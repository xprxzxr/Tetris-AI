import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
import argparse
import multiprocessing as mp
import threading
import time
import json
import torch
import pynvml
import psutil
from flask import Flask, jsonify, send_from_directory
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
from worker import WorkerPool
from collections import deque


# ═══════════════════════════════════════════════════════════════════
#  GPU Governor
# ═══════════════════════════════════════════════════════════════════

class GPUGovernor:
    '''GPU throttle with fixed training parameters and hard utilization ceiling.
    Uses NVML to monitor real GPU utilization and temperature.'''

    def __init__(self, target_high=0.65, burst=3, cooldown=0.05, batch_size=2048,
                 clock_min=None, clock_max=None, mem_clock_min=None, mem_clock_max=None):
        self.target_high = target_high
        self.burst = burst
        self.cooldown = cooldown
        self.batch_size = batch_size

        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
        if isinstance(self._gpu_name, bytes):
            self._gpu_name = self._gpu_name.decode()

        self._clock_locked = False
        self._mem_clock_locked = False
        if clock_min is not None or clock_max is not None:
            self._set_clock_range(clock_min, clock_max)
        if mem_clock_min is not None or mem_clock_max is not None:
            self._set_mem_clock_range(mem_clock_min, mem_clock_max)

        self._ema_util = 0.0
        self._ema_temp = 0.0
        self._alpha = 0.5
        self._peak_util = 0.0
        self._peak_temp = 0.0

    def _set_clock_range(self, clock_min=None, clock_max=None):
        '''Lock GPU core clock to a range (MHz). Requires admin/root.'''
        try:
            min_supported = pynvml.nvmlDeviceGetMinMaxClockOfPState(
                self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS, pynvml.NVML_PSTATE_0)
            lo, hi = min_supported
            if clock_min is None:
                clock_min = lo
            if clock_max is None:
                clock_max = hi
            clock_min = max(lo, min(clock_min, hi))
            clock_max = max(clock_min, min(clock_max, hi))
            pynvml.nvmlDeviceSetGpuLockedClocks(self.gpu_handle, clock_min, clock_max)
            self._clock_locked = True
            print(f"[GPU] Clock locked to {clock_min}-{clock_max} MHz "
                  f"(supported: {lo}-{hi} MHz)")
        except pynvml.NVMLError as e:
            print(f"[GPU] Clock lock failed (need admin/root): {e}")
        except Exception as e:
            print(f"[GPU] Clock lock not supported: {e}")

    def _set_mem_clock_range(self, mem_min=None, mem_max=None):
        '''Lock GPU memory clock to a range (MHz). Requires admin/root.'''
        try:
            min_supported = pynvml.nvmlDeviceGetMinMaxClockOfPState(
                self.gpu_handle, pynvml.NVML_CLOCK_MEM, pynvml.NVML_PSTATE_0)
            lo, hi = min_supported
            if mem_min is None:
                mem_min = lo
            if mem_max is None:
                mem_max = hi
            mem_min = max(lo, min(mem_min, hi))
            mem_max = max(mem_min, min(mem_max, hi))
            pynvml.nvmlDeviceSetMemoryLockedClocks(self.gpu_handle, mem_min, mem_max)
            self._mem_clock_locked = True
            print(f"[GPU] Memory clock locked to {mem_min}-{mem_max} MHz "
                  f"(supported: {lo}-{hi} MHz)")
        except pynvml.NVMLError as e:
            print(f"[GPU] Memory clock lock failed (need admin/root): {e}")
        except Exception as e:
            print(f"[GPU] Memory clock lock not supported: {e}")

    def _reset_clocks(self):
        if self._clock_locked:
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(self.gpu_handle)
                print("[GPU] Core clocks unlocked")
                self._clock_locked = False
            except Exception:
                pass
        if self._mem_clock_locked:
            try:
                pynvml.nvmlDeviceResetMemoryLockedClocks(self.gpu_handle)
                print("[GPU] Memory clocks unlocked")
                self._mem_clock_locked = False
            except Exception:
                pass

    def read_gpu_util(self):
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.gpu / 100.0
        except Exception:
            return self._ema_util

    def read_gpu_temp(self):
        try:
            return pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            return 0

    def read_power(self):
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
        except Exception:
            return 0.0

    def read_clock(self):
        try:
            return pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            return 0

    def read_mem(self):
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return info.used / (1024**3), info.total / (1024**3)
        except Exception:
            return 0.0, 0.0

    def update(self):
        raw_util = self.read_gpu_util()
        raw_temp = self.read_gpu_temp()
        self._ema_util = self._alpha * raw_util + (1 - self._alpha) * self._ema_util
        self._ema_temp = self._alpha * raw_temp + (1 - self._alpha) * self._ema_temp
        self._peak_util = max(self._peak_util, raw_util)
        self._peak_temp = max(self._peak_temp, raw_temp)

    def at_ceiling(self):
        return self.read_gpu_util() >= self.target_high

    def status(self):
        return (f"gpu={self._ema_util:.0%}(pk:{self._peak_util:.0%}) "
                f"{self._ema_temp:.0f}C(pk:{self._peak_temp:.0f}C) "
                f"{self.read_power():.0f}W {self.read_clock()}MHz")

    def shutdown(self):
        self._reset_clocks()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  Web Dashboard (Flask API + static HTML)
# ═══════════════════════════════════════════════════════════════════

DASHBOARD_PORT = int(os.environ.get('DASHBOARD_PORT', 8080))
HISTORY_LEN = 500

# Shared state between training loop and web server
dashboard_data = {
    'episode': 0, 'total_episodes': 0,
    'avg_score': 0, 'best_score': 0, 'epsilon': 1.0,
    'memory': 0, 'gpu_passes': 0,
    'gpu_util': 0, 'gpu_peak_util': 0,
    'gpu_temp': 0, 'gpu_peak_temp': 0,
    'gpu_power': 0, 'gpu_clock': 0,
    'gpu_mem_used': 0, 'gpu_mem_total': 0,
    'gpu_name': '',
    'start_time': 0,
}

# Chart history (append-only from training thread, read by Flask)
chart_history = {
    'episodes': deque(maxlen=HISTORY_LEN),
    'avg_scores': deque(maxlen=HISTORY_LEN),
    'best_scores': deque(maxlen=HISTORY_LEN),
    'gpu_utils': deque(maxlen=HISTORY_LEN),
    'gpu_temps': deque(maxlen=HISTORY_LEN),
    'gpu_powers': deque(maxlen=HISTORY_LEN),
    'timestamps': deque(maxlen=HISTORY_LEN),
}

app = Flask(__name__, static_folder='static')

# Suppress Flask request logging
import logging
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/stats')
def api_stats():
    d = dict(dashboard_data)
    elapsed = time.time() - d.get('start_time', time.time())
    d['elapsed'] = elapsed
    return jsonify(d)


@app.route('/api/history')
def api_history():
    return jsonify({
        'episodes': list(chart_history['episodes']),
        'avg_scores': list(chart_history['avg_scores']),
        'best_scores': list(chart_history['best_scores']),
        'gpu_utils': list(chart_history['gpu_utils']),
        'gpu_temps': list(chart_history['gpu_temps']),
        'gpu_powers': list(chart_history['gpu_powers']),
        'timestamps': list(chart_history['timestamps']),
    })


def _start_web_server():
    app.run(host='0.0.0.0', port=DASHBOARD_PORT, threaded=True)


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def dqn(resume_from=None, fast_mode=False):
    # ── CPU affinity cap ──────────────────────────────────────
    total_cores = psutil.cpu_count(logical=True)
    if fast_mode:
        allowed_cores = list(range(total_cores))
    else:
        half = max(2, total_cores // 2)
        allowed_cores = list(range(half))
    proc = psutil.Process()
    proc.cpu_affinity(allowed_cores)
    print(f"[CPU] Pinned to {len(allowed_cores)}/{total_cores} cores: {allowed_cores}")

    if not fast_mode:
        half_cores = max(2, total_cores // 2)
        torch.set_num_threads(half_cores)
        os.environ['OMP_NUM_THREADS'] = str(half_cores)
        os.environ['MKL_NUM_THREADS'] = str(half_cores)

    # ── Resource caps ──────────────────────────────────────────
    if fast_mode:
        NUM_WORKERS = min(int(mp.cpu_count() * 0.9), 16)
        STEP_THROTTLE = 0
        print("[FAST MODE] Max CPU, no throttle")
    else:
        NUM_WORKERS = min(int(mp.cpu_count() * 0.6), 8)
        STEP_THROTTLE = 0.001
    EPISODES_PER_WORKER = 3

    print(f"[CPU] {NUM_WORKERS} workers x {EPISODES_PER_WORKER} eps "
          f"= {NUM_WORKERS * EPISODES_PER_WORKER} eps/round")

    env = Tetris()
    episodes = 50000
    max_steps = None
    epsilon_stop_episode = 35000  # Explore for 70% of training
    mem_size = 100000
    discount = 0.95  # Focus on near-term rewards (line clears)
    batch_size = 2048
    render_every = None  # No rendering in Docker (no display)
    render_delay = None
    log_every = 50
    replay_start_size = 5000  # Richer initial experience buffer
    n_neurons = [512, 512, 256]
    activations = ['relu', 'relu', 'relu', 'linear']
    lr = 1e-3
    save_best_model = True

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, epsilon_min=0.01,
                     mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size, lr=lr)

    # ── Resume from checkpoint ─────────────────────────────────
    scores = []
    best_score = 0
    start_episode = 0

    if resume_from and os.path.exists(resume_from):
        start_episode, best_score = agent.load_checkpoint(resume_from)
        episodes = start_episode + episodes
        print(f"[RESUME] Loaded checkpoint: episode={start_episode}, "
              f"best_score={best_score}, epsilon={agent.epsilon:.4f}, "
              f"train_steps={agent._train_steps}")
        print(f"[RESUME] Will train {episodes - start_episode} more episodes "
              f"(up to {episodes} total)")
    elif resume_from:
        print(f"[WARN] Checkpoint '{resume_from}' not found, starting fresh")

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-workers={NUM_WORKERS}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    episode = start_episode
    pbar = tqdm(total=episodes, initial=start_episode, desc="Training")

    init_weights = agent.get_weights()
    pool = WorkerPool(NUM_WORKERS, init_weights, step_throttle=STEP_THROTTLE)

    # GPU governor
    if fast_mode:
        governor = GPUGovernor(target_high=1.0, burst=50, cooldown=0.0,
                               batch_size=batch_size,
                               clock_min=None, clock_max=None,
                               mem_clock_min=None, mem_clock_max=None)
    else:
        governor = GPUGovernor(target_high=0.65, burst=20, cooldown=0.01,
                               batch_size=batch_size,
                               clock_min=None, clock_max=None,
                               mem_clock_min=None, mem_clock_max=None)

    # Init dashboard shared state
    dashboard_data['total_episodes'] = episodes
    dashboard_data['start_time'] = time.time()
    dashboard_data['gpu_name'] = governor._gpu_name

    # Start web dashboard
    print(f"[Dashboard] http://0.0.0.0:{DASHBOARD_PORT}")
    web_thread = threading.Thread(target=_start_web_server, daemon=True)
    web_thread.start()

    # GPU training thread
    weights_lock = threading.Lock()
    gpu_shutdown = threading.Event()
    gpu_passes = [0]

    def _gpu_train_loop():
        while not gpu_shutdown.is_set():
            n = agent._mem_count
            bs = governor.batch_size
            if n < replay_start_size or n < bs:
                time.sleep(0.01)
                continue

            if not fast_mode:
                while not gpu_shutdown.is_set():
                    if not governor.at_ceiling():
                        break
                    time.sleep(0.05)

            for _ in range(governor.burst):
                if gpu_shutdown.is_set():
                    return
                agent.train(batch_size=bs, epochs=1)
                gpu_passes[0] += 1

            if not fast_mode:
                torch.cuda.synchronize()
                time.sleep(governor.cooldown)
            governor.update()

            # Push GPU metrics to dashboard every burst
            mem_used, mem_total = governor.read_mem()
            dashboard_data.update({
                'gpu_util': governor._ema_util,
                'gpu_peak_util': governor._peak_util,
                'gpu_temp': governor._ema_temp,
                'gpu_peak_temp': governor._peak_temp,
                'gpu_power': governor.read_power(),
                'gpu_clock': governor.read_clock(),
                'gpu_mem_used': round(mem_used, 2),
                'gpu_mem_total': round(mem_total, 2),
                'gpu_passes': gpu_passes[0],
            })
            # Append to GPU chart history
            chart_history['gpu_utils'].append(round(governor._ema_util, 4))
            chart_history['gpu_temps'].append(round(governor._ema_temp, 1))
            chart_history['gpu_powers'].append(round(governor.read_power(), 1))
            chart_history['timestamps'].append(round(time.time() - dashboard_data['start_time'], 1))

    gpu_thread = threading.Thread(target=_gpu_train_loop, daemon=True)
    gpu_thread.start()

    try:
        # ── Staggered initial dispatch ─────────────────────────────
        for i in range(NUM_WORKERS):
            pool.dispatch_one(init_weights, agent.epsilon, EPISODES_PER_WORKER)
            if i < NUM_WORKERS - 1:
                time.sleep(0.2 / NUM_WORKERS)

        # ── Rolling collection loop ────────────────────────────────
        while episode < episodes:
            all_experiences, ep_scores, ep_steps = pool.collect_one()

            agent.add_batch_to_memory(all_experiences)
            agent.flush_to_gpu()

            with weights_lock:
                weights = agent.get_weights()
            pool.update_weights(weights)
            pool.dispatch_one(weights, agent.epsilon, EPISODES_PER_WORKER)

            scores.extend(ep_scores)
            round_count = len(ep_scores)
            episode += round_count
            pbar.update(round_count)

            for s in ep_scores:
                if save_best_model and s > best_score:
                    print(f'\nNew best: score={s}, ep={episode} '
                          f'(GPU passes so far: {gpu_passes[0]})')
                    best_score = s
                    with weights_lock:
                        agent.save_model("best.pt")

            # Dashboard + chart history
            recent_avg = mean(scores[-50:]) if scores else 0
            dashboard_data.update({
                'episode': episode,
                'avg_score': round(recent_avg, 2),
                'best_score': best_score,
                'epsilon': round(agent.epsilon, 6),
                'memory': agent._mem_len(),
            })
            chart_history['episodes'].append(episode)
            chart_history['avg_scores'].append(round(recent_avg, 2))
            chart_history['best_scores'].append(best_score)

            if log_every and episode >= log_every and episode % log_every < round_count:
                avg_score = mean(scores[-log_every:])
                min_score = min(scores[-log_every:])
                max_score = max(scores[-log_every:])
                log.log(episode, avg_score=avg_score, min_score=min_score,
                        max_score=max_score)
                pbar.set_postfix_str(
                    f"avg={avg_score:.0f} best={best_score:.0f} "
                    f"eps={agent.epsilon:.3f} mem={agent._mem_len()} "
                    f"passes={gpu_passes[0]} | {governor.status()}"
                )

            # ── Auto-save checkpoint every 500 episodes ────────────
            if episode % 500 < round_count:
                with weights_lock:
                    agent.save_checkpoint("checkpoint.pt", episode, best_score)

    finally:
        gpu_shutdown.set()
        gpu_thread.join(timeout=5)
        governor.shutdown()
        pool.shutdown()

    # Final checkpoint save
    agent.save_checkpoint("checkpoint.pt", episode, best_score)
    print(f"[SAVE] Checkpoint saved: episode={episode}, best_score={best_score}")

    pbar.close()
    print(f"\nDone. Best score: {best_score}, GPU training passes: {gpu_passes[0]}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file (e.g. checkpoint.pt)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: max CPU, no throttle')
    args = parser.parse_args()
    dqn(resume_from=args.resume, fast_mode=args.fast)
