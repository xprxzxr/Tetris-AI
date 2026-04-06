'''
Parallel Tetris workers with shared memory weights + multi-episode batches.
Workers stay alive, reuse envs, and pull weights from shared memory
instead of deserializing from Queue each time.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
import time as _time
import torch
import torch.nn as nn
from tetris import Tetris
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
import struct

# Per-step sleep to spread CPU load evenly (prevents spikes)
# Set to 0 to disable throttling (fast mode)
STEP_THROTTLE = 0.001  # 1ms pause between game steps — overridden by WorkerPool


def _build_model(weights_dict):
    '''Build model from weight dict (CPU only)'''
    keys = sorted([k for k in weights_dict if 'weight' in k])
    layers = []
    for i, k in enumerate(keys):
        w = weights_dict[k]
        in_f, out_f = w.shape[1], w.shape[0]
        layers.append(nn.Linear(in_f, out_f))
        if i < len(keys) - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    _load_weights(model, weights_dict)
    model.eval()
    return model


def _load_weights(model, weights_dict):
    '''Load weights into existing model, stripping net. prefix'''
    state_dict = {}
    for k, v in weights_dict.items():
        state_dict[k.replace('net.', '')] = torch.tensor(v)
    model.load_state_dict(state_dict)


def _load_weights_from_shm(model, shm_name, param_layout, version_box):
    '''Load weights from shared memory block. Skips if version unchanged.'''
    shm = SharedMemory(name=shm_name, create=False)
    # First 8 bytes = uint64 version counter
    new_version = struct.unpack('Q', shm.buf[:8])[0]
    if new_version == version_box[0]:
        shm.close()
        return  # Weights haven't changed
    version_box[0] = new_version

    state_dict = {}
    for clean_key, offset, shape, nbytes in param_layout:
        arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf, offset=offset).copy()
        state_dict[clean_key] = torch.from_numpy(arr)
    model.load_state_dict(state_dict)
    shm.close()


def _worker_loop(task_queue, result_queue, shm_name, param_layout, step_throttle=0.001):
    '''Persistent worker with shared memory weight loading + multi-episode.
    Returns results one episode at a time for rolling collection.'''
    model = None
    env = Tetris()
    version_box = [-1]

    while True:
        task = task_queue.get()
        if task is None:
            break

        init_weights, epsilon, episodes_per_worker = task

        if model is None:
            if init_weights is None:
                # First task came without weights — build from SharedMemory
                # Read shapes from param_layout to construct the model
                fake_weights = {}
                for clean_key, offset, shape, nbytes in param_layout:
                    fake_weights[clean_key] = np.zeros(shape, dtype=np.float32)
                model = _build_model(fake_weights)
            else:
                model = _build_model(init_weights)
            version_box[0] = -1

        _load_weights_from_shm(model, shm_name, param_layout, version_box)

        all_experiences = []
        all_scores = []
        all_steps = []

        for _ in range(episodes_per_worker):
            current_state = env.reset()
            done = False
            steps = 0
            experiences = []

            while not done:
                next_states = {tuple(v): k for k, v in env.get_next_states().items()}

                if not next_states:
                    break

                if random.random() <= epsilon:
                    best_state = random.choice(list(next_states.keys()))
                else:
                    states_list = list(next_states.keys())
                    with torch.no_grad():
                        states_t = torch.tensor(states_list, dtype=torch.float32)
                        values = model(states_t)
                        best_idx = torch.argmax(values).item()
                    best_state = states_list[best_idx]

                best_action = next_states[best_state]
                reward, done = env.play(best_action[0], best_action[1])

                experiences.append((current_state, best_state, reward, done))
                current_state = best_state
                steps += 1

                # Throttle CPU — small sleep spreads load evenly
                if step_throttle > 0:
                    _time.sleep(step_throttle)

            all_experiences.extend(experiences)
            all_scores.append(env.get_game_score())
            all_steps.append(steps)

        result_queue.put((all_experiences, all_scores, all_steps))


class WorkerPool:
    '''Pool of persistent workers with shared memory weight transport.'''

    def __init__(self, num_workers, init_weights, step_throttle=0.001):
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []

        # Set up shared memory for weights
        self._param_layout = []  # [(clean_key, offset, shape, nbytes), ...]
        offset = 8  # First 8 bytes reserved for version counter
        for k, v in init_weights.items():
            clean_key = k.replace('net.', '')
            arr = np.asarray(v, dtype=np.float32)
            nbytes = arr.nbytes
            self._param_layout.append((clean_key, offset, arr.shape, nbytes))
            offset += nbytes

        self._shm_size = offset
        self._shm = SharedMemory(create=True, size=self._shm_size)
        self._shm_name = self._shm.name
        self._version = 0

        # Write initial weights
        self._write_weights(init_weights)

        # Start workers
        for _ in range(num_workers):
            p = Process(target=_worker_loop,
                        args=(self.task_queue, self.result_queue,
                              self._shm_name, self._param_layout, step_throttle),
                        daemon=True)
            p.start()
            self.workers.append(p)

    def _write_weights(self, weights):
        '''Write weights to shared memory + bump version.'''
        self._version += 1
        struct.pack_into('Q', self._shm.buf, 0, self._version)
        for clean_key, offset, shape, nbytes in self._param_layout:
            orig_key = 'net.' + clean_key
            arr = np.asarray(weights[orig_key], dtype=np.float32)
            self._shm.buf[offset:offset + nbytes] = arr.tobytes()

    def update_weights(self, weights):
        '''Update shared memory with new weights (called from main process).'''
        self._write_weights(weights)

    def dispatch(self, init_weights, epsilon, episodes_per_worker):
        '''Dispatch work to all workers. Non-blocking — call collect() to get results.'''
        for _ in range(self.num_workers):
            # Only send weights on first call (workers use SharedMemory after that)
            self.task_queue.put((init_weights, epsilon, episodes_per_worker))
        self._pending = self.num_workers

    def dispatch_one(self, init_weights, epsilon, episodes_per_worker):
        '''Dispatch a single task (for rolling re-dispatch).
        Sends None for weights to avoid serializing large dicts through Queue —
        workers read weights from SharedMemory instead.'''
        self.task_queue.put((None, epsilon, episodes_per_worker))
        self._pending = getattr(self, '_pending', 0) + 1

    def collect_one(self):
        '''Collect one result. Blocking until one worker finishes.'''
        result = self.result_queue.get()
        self._pending = max(0, getattr(self, '_pending', 1) - 1)
        return result

    def collect(self):
        '''Collect results from all workers. Blocking.'''
        results = []
        for _ in range(getattr(self, '_pending', self.num_workers)):
            results.append(self.result_queue.get())
        self._pending = 0
        return results

    def run_episodes(self, init_weights, epsilon, episodes_per_worker):
        '''Dispatch + collect in one call.'''
        self.dispatch(init_weights, epsilon, episodes_per_worker)
        return self.collect()

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join(timeout=5)
        self._shm.close()
        self._shm.unlink()
