'''
Parallel Tetris workers with GPU-batched inference.
Each worker runs N simultaneous games and batches their GPU inference
into a single request per step — amortizing queue latency.
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

GAMES_PER_WORKER = 4  # Simultaneous games per worker process


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


def _worker_loop(task_queue, result_queue, inference_queue, response_queues, worker_id,
                 step_throttle=0.0, n_step=3, discount=0.95):
    '''Persistent worker — runs N simultaneous Tetris games.
    Batches all games\' GPU inference into one request per step.'''
    n_games = GAMES_PER_WORKER
    envs = [Tetris() for _ in range(n_games)]
    my_response_queue = response_queues[worker_id]

    while True:
        task = task_queue.get()
        if task is None:
            break

        epsilon, total_episodes = task

        all_experiences = []
        all_scores = []
        all_steps = []
        episodes_done = 0

        # Per-game state tracking
        current_states = [None] * n_games
        game_experiences = [[] for _ in range(n_games)]
        game_steps = [0] * n_games

        # Reset all games
        for g in range(n_games):
            current_states[g] = envs[g].reset()

        # Step all games simultaneously until we've collected enough episodes
        while episodes_done < total_episodes:
            # Phase 1: Generate next states for all active games
            per_game_data = []  # (game_idx, next_states_dict, states_list, state_to_action)
            for g in range(n_games):
                if episodes_done >= total_episodes:
                    break
                next_states_dict = envs[g].get_next_states()
                if not next_states_dict:
                    # Game over — no valid moves
                    game_experiences[g].append(
                        (current_states[g], current_states[g], -5.0, True))
                    # Finish episode
                    all_experiences.extend(
                        _compute_nstep(game_experiences[g], n_step, discount))
                    all_scores.append(envs[g].get_game_score())
                    all_steps.append(game_steps[g])
                    episodes_done += 1
                    # Reset this game
                    current_states[g] = envs[g].reset()
                    game_experiences[g] = []
                    game_steps[g] = 0
                    continue

                state_to_action = {}
                states_list = []
                for action, state_vec in next_states_dict.items():
                    state_tuple = tuple(state_vec)
                    state_to_action[state_tuple] = action
                    states_list.append(state_vec)

                per_game_data.append((g, next_states_dict, states_list, state_to_action))

            if not per_game_data:
                continue

            # Phase 2: Decide actions — batch GPU inference for all games
            # Separate random (epsilon) games from GPU games
            random_games = []
            gpu_games = []
            for item in per_game_data:
                if random.random() <= epsilon:
                    random_games.append(item)
                else:
                    gpu_games.append(item)

            # Handle random selections locally
            game_actions = {}  # game_idx → (best_state, best_action)
            for g, nsd, sl, sta in random_games:
                best_state = random.choice(list(sta.keys()))
                best_action = sta[best_state]
                game_actions[g] = (best_state, best_action)

            # Batch GPU inference for remaining games
            if gpu_games:
                all_states = []
                game_counts = []
                for g, nsd, sl, sta in gpu_games:
                    all_states.extend(sl)
                    game_counts.append(len(sl))

                states_array = np.array(all_states, dtype=np.float32)
                inference_queue.put((worker_id, states_array, game_counts))
                best_indices = my_response_queue.get()

                # Map GPU results back to per-game actions
                for i, (g, nsd, sl, sta) in enumerate(gpu_games):
                    best_idx = best_indices[i]
                    best_state = tuple(sl[best_idx])
                    best_action = sta[best_state]
                    game_actions[g] = (best_state, best_action)

            # Phase 3: Execute actions for all games
            for g, nsd, sl, sta in per_game_data:
                if g not in game_actions:
                    continue
                best_state, best_action = game_actions[g]
                reward, done = envs[g].play(best_action[0], best_action[1])

                game_experiences[g].append(
                    (current_states[g], best_state, reward, done))
                current_states[g] = best_state
                game_steps[g] += 1

                if done:
                    # Episode finished
                    all_experiences.extend(
                        _compute_nstep(game_experiences[g], n_step, discount))
                    all_scores.append(envs[g].get_game_score())
                    all_steps.append(game_steps[g])
                    episodes_done += 1
                    # Reset this game
                    current_states[g] = envs[g].reset()
                    game_experiences[g] = []
                    game_steps[g] = 0

            if step_throttle > 0:
                _time.sleep(step_throttle)

        result_queue.put((all_experiences, all_scores, all_steps))


class WorkerPool:
    '''Pool of persistent workers with GPU-batched inference.
    Each worker runs N simultaneous games and batches inference.'''

    def __init__(self, num_workers, step_throttle=0.0, n_step=3, discount=0.95):
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.result_queue = Queue()

        # GPU inference queues
        self.inference_queue = Queue()  # Workers → GPU: (worker_id, states_array, game_counts)
        self.response_queues = [Queue() for _ in range(num_workers)]  # GPU → Workers: best_indices list

        self.workers = []
        for i in range(num_workers):
            p = Process(target=_worker_loop,
                        args=(self.task_queue, self.result_queue,
                              self.inference_queue, self.response_queues, i,
                              step_throttle, n_step, discount),
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
