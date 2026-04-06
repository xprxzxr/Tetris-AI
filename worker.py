'''
Parallel Tetris workers with LOCAL CPU models for inference.
Workers run their own lightweight CPU model copy for action selection.
Weights are synced from the GPU training model periodically via shared memory.
No GPU queue bottleneck — workers never block on GPU.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory


def _compute_nstep(experiences, n_step, discount):
    '''Convert 1-step experiences to n-step returns. CPU-side, per episode.'''
    if n_step <= 1:
        return experiences
    n = len(experiences)
    result = []
    for i in range(n):
        R = 0.0
        gamma_power = 1.0
        end_idx = i
        for k in range(n_step):
            idx = i + k
            if idx >= n:
                break
            _, _, r_k, d_k = experiences[idx]
            R += gamma_power * r_k
            gamma_power *= discount
            end_idx = idx
            if d_k:
                break
        s_i = experiences[i][0]
        s_end = experiences[end_idx][1]
        done_end = experiences[end_idx][3]
        result.append((s_i, s_end, R, done_end))
    return result


def _build_model(state_size, n_neurons, activations):
    '''Build a lightweight CPU model matching the GPU model architecture.'''
    import torch.nn as nn
    layers = []
    in_size = state_size
    for i, neurons in enumerate(n_neurons):
        layers.append(nn.Linear(in_size, neurons))
        act = activations[i]
        if act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'tanh':
            layers.append(nn.Tanh())
        in_size = neurons
    layers.append(nn.Linear(in_size, 1))
    if activations[-1] == 'relu':
        layers.append(nn.ReLU())
    elif activations[-1] == 'tanh':
        layers.append(nn.Tanh())
    model = nn.Sequential(*layers)
    model.eval()
    return model


def _load_weights_from_shm(model, shm_name, weight_shapes, weight_keys):
    '''Load model weights from shared memory block.'''
    import torch
    try:
        shm = SharedMemory(name=shm_name, create=False)
        offset = 0
        state_dict = {}
        for key, shape in zip(weight_keys, weight_shapes):
            size = 1
            for s in shape:
                size *= s
            arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf, offset=offset * 4)
            state_dict[key] = torch.from_numpy(arr.copy())
            offset += size
        model.load_state_dict(state_dict)
        shm.close()
        return True
    except Exception:
        return False


def _worker_loop(task_queue, result_queue, worker_id, shm_name,
                 weight_shapes, weight_keys, state_size, n_neurons, activations,
                 n_step=3, discount=0.95):
    '''Persistent worker with local CPU model. No GPU dependency for inference.'''
    import torch
    # Each worker gets exactly 1 thread — prevents 28 workers × N threads fighting for cores
    torch.set_num_threads(1)
    from tetris import Tetris
    env = Tetris()
    model = _build_model(state_size, n_neurons, activations)

    # Initial weight load
    _load_weights_from_shm(model, shm_name, weight_shapes, weight_keys)

    while True:
        task = task_queue.get()
        if task is None:
            break

        epsilon, episodes_per_worker, sync_weights = task

        # Sync weights from shared memory if flagged
        if sync_weights:
            _load_weights_from_shm(model, shm_name, weight_shapes, weight_keys)

        all_experiences = []
        all_scores = []
        all_steps = []

        for _ in range(episodes_per_worker):
            current_state = env.reset()
            done = False
            steps = 0
            experiences = []

            while not done:
                next_states_dict = env.get_next_states()

                if not next_states_dict:
                    break

                actions = list(next_states_dict.keys())
                states = list(next_states_dict.values())

                if random.random() <= epsilon:
                    idx = random.randint(0, len(actions) - 1)
                else:
                    # Local CPU inference — no GPU queue, no blocking
                    with torch.no_grad():
                        states_np = np.stack(states).astype(np.float32)
                        states_t = torch.from_numpy(states_np)
                        preds = model(states_t)
                        idx = int(torch.argmax(preds).item())

                best_action = actions[idx]
                best_state = tuple(states[idx])
                reward, done = env.play(best_action[0], best_action[1])

                experiences.append((current_state, best_state, reward, done))
                current_state = best_state
                steps += 1

            all_experiences.extend(_compute_nstep(experiences, n_step, discount))
            all_scores.append(env.get_game_score())
            all_steps.append(steps)

        result_queue.put((all_experiences, all_scores, all_steps))


class WorkerPool:
    '''Pool of persistent workers with local CPU models.
    Weights synced from GPU model via shared memory.'''

    def __init__(self, num_workers, init_weights, state_size, n_neurons, activations,
                 n_step=3, discount=0.95):
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.result_queue = Queue()

        # Set up shared memory for weight distribution
        self._weight_keys = list(init_weights.keys())
        self._weight_shapes = [init_weights[k].shape for k in self._weight_keys]
        total_floats = sum(np.prod(s) for s in self._weight_shapes)
        self._shm = SharedMemory(create=True, size=total_floats * 4)
        self._shm_name = self._shm.name
        self._total_floats = total_floats

        # Write initial weights to shared memory
        self._write_weights(init_weights)

        self.workers = []
        for i in range(num_workers):
            p = Process(target=_worker_loop,
                        args=(self.task_queue, self.result_queue, i,
                              self._shm_name, self._weight_shapes, self._weight_keys,
                              state_size, n_neurons, activations,
                              n_step, discount),
                        daemon=True)
            p.start()
            self.workers.append(p)

    def _write_weights(self, weights):
        '''Write weight dict to shared memory.'''
        offset = 0
        for key in self._weight_keys:
            arr = weights[key]
            flat = arr.flatten().astype(np.float32)
            dest = np.ndarray(flat.shape, dtype=np.float32,
                              buffer=self._shm.buf, offset=offset * 4)
            dest[:] = flat
            offset += len(flat)

    def update_weights(self, weights):
        '''Update shared memory with new weights from GPU model.'''
        self._write_weights(weights)

    def dispatch_one(self, epsilon, episodes_per_worker, sync_weights=False):
        '''Dispatch a single task to one worker.'''
        self.task_queue.put((epsilon, episodes_per_worker, sync_weights))
        self._pending = getattr(self, '_pending', 0) + 1

    def collect_one(self):
        '''Collect one result. Blocking until one worker finishes.'''
        result = self.result_queue.get()
        self._pending = max(0, getattr(self, '_pending', 1) - 1)
        return result

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join(timeout=5)
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
