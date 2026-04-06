'''
Parallel Tetris workers with GPU-batched inference.
Workers run Numba-JIT game logic and send candidate states to GPU
for evaluation instead of running their own CPU models.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
import time as _time
from tetris import Tetris
from multiprocessing import Process, Queue


def _worker_loop(task_queue, result_queue, inference_queue, response_queues, worker_id,
                 step_throttle=0.0):
    '''Persistent worker — plays Tetris games, sends states to GPU for evaluation.
    No local model. All inference happens on GPU via queues.'''
    env = Tetris()
    my_response_queue = response_queues[worker_id]

    while True:
        task = task_queue.get()
        if task is None:
            break

        epsilon, episodes_per_worker = task

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

                # Flip: {action: state_vec} → {state_tuple: action}
                state_to_action = {}
                states_list = []
                for action, state_vec in next_states_dict.items():
                    state_tuple = tuple(state_vec)
                    state_to_action[state_tuple] = action
                    states_list.append(state_vec)

                if random.random() <= epsilon:
                    # Random exploration — no GPU needed
                    best_state = random.choice(list(state_to_action.keys()))
                else:
                    # Send to GPU for batch evaluation
                    states_array = np.array(states_list, dtype=np.float32)
                    inference_queue.put((worker_id, states_array))
                    # Block until GPU returns the best index
                    best_idx = my_response_queue.get()
                    best_state = tuple(states_list[best_idx])

                best_action = state_to_action[best_state]
                reward, done = env.play(best_action[0], best_action[1])

                experiences.append((current_state, best_state, reward, done))
                current_state = best_state
                steps += 1

                if step_throttle > 0:
                    _time.sleep(step_throttle)

            all_experiences.extend(experiences)
            all_scores.append(env.get_game_score())
            all_steps.append(steps)

        result_queue.put((all_experiences, all_scores, all_steps))


class WorkerPool:
    '''Pool of persistent workers with GPU-batched inference.
    Workers send candidate states to GPU via inference_queue.
    GPU inference thread (in run.py) evaluates and returns results.'''

    def __init__(self, num_workers, step_throttle=0.0):
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.result_queue = Queue()

        # GPU inference queues
        self.inference_queue = Queue()  # Workers → GPU: (worker_id, states_array)
        self.response_queues = [Queue() for _ in range(num_workers)]  # GPU → Workers: best_idx

        self.workers = []
        for i in range(num_workers):
            p = Process(target=_worker_loop,
                        args=(self.task_queue, self.result_queue,
                              self.inference_queue, self.response_queues, i,
                              step_throttle),
                        daemon=True)
            p.start()
            self.workers.append(p)

    def dispatch_one(self, epsilon, episodes_per_worker):
        '''Dispatch a single task to one worker.'''
        self.task_queue.put((epsilon, episodes_per_worker))
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
