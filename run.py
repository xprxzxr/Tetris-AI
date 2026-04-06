import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
import argparse
import multiprocessing as mp
import threading
import time
import torch
import pynvml
import psutil
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
from worker import WorkerPool
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque


# ═══════════════════════════════════════════════════════════════════
#  Live Training Dashboard
# ═══════════════════════════════════════════════════════════════════

class TrainingDashboard:
    '''Real-time training dashboard with charts + stats panel.
    Runs tkinter mainloop in its own daemon thread.
    Reads from a shared data dict written by the training loop.'''

    HISTORY_LEN = 500  # Max data points to keep per chart

    def __init__(self, shared_data, target_high=0.65):
        self.data = shared_data
        self.target_high = target_high
        self._alive = True

        # Chart histories
        self._ep_history = deque(maxlen=self.HISTORY_LEN)
        self._avg_score_history = deque(maxlen=self.HISTORY_LEN)
        self._best_score_history = deque(maxlen=self.HISTORY_LEN)
        self._gpu_util_history = deque(maxlen=self.HISTORY_LEN)
        self._gpu_temp_history = deque(maxlen=self.HISTORY_LEN)
        self._gpu_power_history = deque(maxlen=self.HISTORY_LEN)
        self._time_history = deque(maxlen=self.HISTORY_LEN)
        self._start_time = time.time()

    def run(self):
        '''Entry point — call from a daemon thread.'''
        try:
            self._build_ui()
            self._root.mainloop()
        except Exception:
            pass  # Dashboard crash should never kill training
        finally:
            self._alive = False

    def _build_ui(self):
        self._root = tk.Tk()
        self._root.title("Tetris AI — Training Dashboard")
        self._root.configure(bg='#1e1e1e')
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Try to size the window nicely
        self._root.geometry("1280x720")

        # ── Main layout: left = charts, right = stats ──
        main_frame = tk.Frame(self._root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: matplotlib charts
        chart_frame = tk.Frame(main_frame, bg='#1e1e1e')
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        plt.style.use('dark_background')
        self._fig, (self._ax_score, self._ax_util, self._ax_temp) = plt.subplots(
            3, 1, figsize=(8, 6), facecolor='#1e1e1e')
        self._fig.subplots_adjust(hspace=0.4, left=0.10, right=0.95, top=0.95, bottom=0.08)

        # Score chart
        self._ax_score.set_title('Score', fontsize=10, color='#cccccc')
        self._ax_score.set_facecolor('#2d2d2d')
        self._line_avg, = self._ax_score.plot([], [], color='#4fc3f7', linewidth=1.5, label='Avg')
        self._line_best, = self._ax_score.plot([], [], color='#81c784', linewidth=1, linestyle='--', label='Best')
        self._ax_score.legend(loc='upper left', fontsize=8)
        self._ax_score.tick_params(colors='#888888', labelsize=8)

        # GPU Utilization chart
        self._ax_util.set_title('GPU Utilization', fontsize=10, color='#cccccc')
        self._ax_util.set_facecolor('#2d2d2d')
        self._line_util, = self._ax_util.plot([], [], color='#ffb74d', linewidth=1.5)
        self._ax_util.axhline(y=self.target_high * 100, color='#ef5350', linestyle='--',
                              linewidth=1, alpha=0.7, label=f'Ceiling ({self.target_high:.0%})')
        self._ax_util.set_ylim(0, 100)
        self._ax_util.set_ylabel('%', fontsize=8, color='#888888')
        self._ax_util.legend(loc='upper left', fontsize=8)
        self._ax_util.tick_params(colors='#888888', labelsize=8)

        # GPU Temperature chart
        self._ax_temp.set_title('GPU Temperature / Power', fontsize=10, color='#cccccc')
        self._ax_temp.set_facecolor('#2d2d2d')
        self._line_temp, = self._ax_temp.plot([], [], color='#ef5350', linewidth=1.5, label='Temp')
        self._ax_temp_power = self._ax_temp.twinx()
        self._line_power, = self._ax_temp_power.plot([], [], color='#ce93d8', linewidth=1, alpha=0.7, label='Power')
        self._ax_temp.set_ylabel('°C', fontsize=8, color='#888888')
        self._ax_temp_power.set_ylabel('W', fontsize=8, color='#888888')
        self._ax_temp.set_xlabel('Time (s)', fontsize=8, color='#888888')
        self._ax_temp.legend(loc='upper left', fontsize=8)
        self._ax_temp_power.legend(loc='upper right', fontsize=8)
        self._ax_temp.tick_params(colors='#888888', labelsize=8)
        self._ax_temp_power.tick_params(colors='#888888', labelsize=8)

        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: stats panel
        stats_frame = tk.Frame(main_frame, bg='#252525', width=280)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        stats_frame.pack_propagate(False)

        title_label = tk.Label(stats_frame, text="LIVE STATS", font=("Consolas", 14, "bold"),
                               fg='#4fc3f7', bg='#252525')
        title_label.pack(pady=(15, 10))

        self._stat_labels = {}
        stats_config = [
            ("Episode", "episode"),
            ("Avg Score", "avg_score"),
            ("Best Score", "best_score"),
            ("Epsilon", "epsilon"),
            ("Replay Buffer", "memory"),
            ("", None),  # Spacer
            ("GPU Passes", "gpu_passes"),
            ("GPU Util", "gpu_util"),
            ("GPU Peak Util", "gpu_peak_util"),
            ("GPU Temp", "gpu_temp"),
            ("GPU Peak Temp", "gpu_peak_temp"),
            ("GPU Power", "gpu_power"),
            ("GPU Clock", "gpu_clock"),
            ("", None),  # Spacer
            ("Elapsed", "elapsed"),
        ]

        for label_text, key in stats_config:
            if key is None:
                spacer = tk.Frame(stats_frame, height=10, bg='#252525')
                spacer.pack()
                continue

            row = tk.Frame(stats_frame, bg='#252525')
            row.pack(fill=tk.X, padx=15, pady=2)

            name_lbl = tk.Label(row, text=label_text, font=("Consolas", 9),
                                fg='#888888', bg='#252525', anchor='w')
            name_lbl.pack(side=tk.LEFT)

            val_lbl = tk.Label(row, text="--", font=("Consolas", 11, "bold"),
                               fg='#ffffff', bg='#252525', anchor='e')
            val_lbl.pack(side=tk.RIGHT)
            self._stat_labels[key] = val_lbl

        # Start refresh timer
        self._root.after(1000, self._refresh)

    def _refresh(self):
        '''Called every second to update charts and stats.'''
        if not self._alive:
            return
        try:
            self._update_data()
            self._update_charts()
            self._update_stats()
            self._canvas.draw_idle()
        except Exception:
            pass  # Never crash the dashboard
        if self._alive:
            self._root.after(1000, self._refresh)

    def _update_data(self):
        '''Pull latest data from shared dict into chart histories.'''
        d = self.data
        ep = d.get('episode', 0)
        if ep > 0 and (not self._ep_history or ep != self._ep_history[-1]):
            elapsed = time.time() - self._start_time
            self._ep_history.append(ep)
            self._avg_score_history.append(d.get('avg_score', 0))
            self._best_score_history.append(d.get('best_score', 0))
            self._time_history.append(elapsed)

        # GPU metrics update every refresh regardless
        elapsed = time.time() - self._start_time
        self._gpu_util_history.append(d.get('gpu_util', 0))
        self._gpu_temp_history.append(d.get('gpu_temp', 0))
        self._gpu_power_history.append(d.get('gpu_power', 0))
        if elapsed not in self._time_history:
            pass  # GPU metrics use their own index

    def _update_charts(self):
        '''Redraw chart lines.'''
        if self._ep_history:
            eps = list(self._ep_history)
            self._line_avg.set_data(eps, list(self._avg_score_history))
            self._line_best.set_data(eps, list(self._best_score_history))
            self._ax_score.relim()
            self._ax_score.autoscale_view()

        n = len(self._gpu_util_history)
        if n > 0:
            xs = list(range(n))
            self._line_util.set_data(xs, [u * 100 for u in self._gpu_util_history])
            self._ax_util.set_xlim(0, max(n, 10))

            self._line_temp.set_data(xs, list(self._gpu_temp_history))
            self._line_power.set_data(xs, list(self._gpu_power_history))
            self._ax_temp.relim()
            self._ax_temp.autoscale_view()
            self._ax_temp_power.relim()
            self._ax_temp_power.autoscale_view()

    def _update_stats(self):
        '''Update stat labels from shared data.'''
        d = self.data
        elapsed = time.time() - self._start_time

        updates = {
            'episode': f"{d.get('episode', 0)} / {d.get('total_episodes', 0)}",
            'avg_score': f"{d.get('avg_score', 0):.1f}",
            'best_score': f"{d.get('best_score', 0):.0f}",
            'epsilon': f"{d.get('epsilon', 1.0):.4f}",
            'memory': f"{d.get('memory', 0):,}",
            'gpu_passes': f"{d.get('gpu_passes', 0):,}",
            'gpu_util': f"{d.get('gpu_util', 0):.1%}",
            'gpu_peak_util': f"{d.get('gpu_peak_util', 0):.1%}",
            'gpu_temp': f"{d.get('gpu_temp', 0):.0f} °C",
            'gpu_peak_temp': f"{d.get('gpu_peak_temp', 0):.0f} °C",
            'gpu_power': f"{d.get('gpu_power', 0):.1f} W",
            'gpu_clock': f"{d.get('gpu_clock', 0)} MHz",
            'elapsed': self._format_time(elapsed),
        }

        for key, val in updates.items():
            if key in self._stat_labels:
                self._stat_labels[key].configure(text=val)

                # Color code GPU util
                if key == 'gpu_util':
                    util = d.get('gpu_util', 0)
                    if util >= self.target_high:
                        self._stat_labels[key].configure(fg='#ef5350')  # Red
                    elif util >= self.target_high * 0.8:
                        self._stat_labels[key].configure(fg='#ffb74d')  # Orange
                    else:
                        self._stat_labels[key].configure(fg='#81c784')  # Green

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    def _on_close(self):
        self._alive = False
        try:
            self._root.destroy()
        except Exception:
            pass


class GPUGovernor:
    '''GPU throttle with fixed training parameters and hard utilization ceiling.
    Uses NVML to monitor real GPU utilization and temperature.
    Training is gated — will not run if GPU is at or above target_high.'''

    def __init__(self, target_high=0.65, burst=3, cooldown=0.05, batch_size=2048,
                 clock_min=1600, clock_max=2000, mem_clock_min=9000, mem_clock_max=9500):
        self.target_high = target_high
        self.burst = burst
        self.cooldown = cooldown
        self.batch_size = batch_size

        # NVML setup
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Clock locking — pin GPU to a specific frequency range
        self._clock_locked = False
        self._mem_clock_locked = False
        if clock_min is not None or clock_max is not None:
            self._set_clock_range(clock_min, clock_max)
        if mem_clock_min is not None or mem_clock_max is not None:
            self._set_mem_clock_range(mem_clock_min, mem_clock_max)

        # Thermal tracking
        self._ema_util = 0.0
        self._ema_temp = 0.0
        self._alpha = 0.5  # EMA smoothing
        self._peak_util = 0.0
        self._peak_temp = 0.0

    def _set_clock_range(self, clock_min=None, clock_max=None):
        '''Lock GPU core clock to a range (MHz). Requires admin/root.
        Example: clock_min=800, clock_max=1500 keeps GPU between 800-1500 MHz.
        Set both to the same value for a perfectly flat clock line.'''
        try:
            # Query supported clock range
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
            print(f"[GPU] Run as Administrator to enable clock locking")
        except Exception as e:
            print(f"[GPU] Clock lock not supported: {e}")

    def _set_mem_clock_range(self, mem_min=None, mem_max=None):
        '''Lock GPU memory clock to a range (MHz). Requires admin/root.
        Example: mem_min=5001, mem_max=5001 locks VRAM to 5001 MHz.'''
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
        '''Unlock all GPU clocks back to default on shutdown.'''
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
        '''Read current GPU utilization (0.0 - 1.0).'''
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.gpu / 100.0
        except Exception:
            return self._ema_util

    def read_gpu_temp(self):
        '''Read current GPU temperature in °C.'''
        try:
            return pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            return 0

    def read_power(self):
        '''Read current GPU power draw in watts.'''
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
        except Exception:
            return 0.0

    def read_clock(self):
        '''Read current GPU clock speed in MHz.'''
        try:
            return pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            return 0

    def update(self):
        '''Update smoothed metrics. Call after each burst cycle.'''
        raw_util = self.read_gpu_util()
        raw_temp = self.read_gpu_temp()
        self._ema_util = self._alpha * raw_util + (1 - self._alpha) * self._ema_util
        self._ema_temp = self._alpha * raw_temp + (1 - self._alpha) * self._ema_temp
        self._peak_util = max(self._peak_util, raw_util)
        self._peak_temp = max(self._peak_temp, raw_temp)

    def at_ceiling(self):
        '''Returns True if GPU is at or above the hard ceiling.'''
        return self.read_gpu_util() >= self.target_high

    def status(self):
        '''Return current thermals for logging.'''
        return (f"gpu={self._ema_util:.0%}(pk:{self._peak_util:.0%}) "
                f"{self._ema_temp:.0f}°C(pk:{self._peak_temp:.0f}°C) "
                f"{self.read_power():.0f}W {self.read_clock()}MHz")

    def shutdown(self):
        self._reset_clocks()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def dqn(resume_from=None, fast_mode=False):
    # ── CPU affinity cap ──────────────────────────────────────
    # Pin this process (and all child workers) to a fraction of CPU cores.
    # This is the ONLY reliable way to cap system CPU usage —
    # sleep-based throttling can't control PyTorch's internal thread pool.
    total_cores = psutil.cpu_count(logical=True)
    if fast_mode:
        # Fast mode: use all cores
        allowed_cores = list(range(total_cores))
    else:
        # Normal mode: use ~70% of cores → ~70% system CPU
        cap = max(2, int(total_cores * 0.7))
        allowed_cores = list(range(cap))
    proc = psutil.Process()
    proc.cpu_affinity(allowed_cores)
    print(f"[CPU] Pinned to {len(allowed_cores)}/{total_cores} cores: {allowed_cores}")

    # Also limit PyTorch's internal thread pool (MKL, OpenMP, etc.)
    if not fast_mode:
        pt_threads = max(2, int(total_cores * 0.7))
        torch.set_num_threads(pt_threads)
        os.environ['OMP_NUM_THREADS'] = str(pt_threads)
        os.environ['MKL_NUM_THREADS'] = str(pt_threads)
        print(f"[CPU] PyTorch threads capped at {pt_threads}")

    # ── Resource caps ──────────────────────────────────────────
    if fast_mode:
        NUM_WORKERS = max(4, mp.cpu_count() - 2)
        STEP_THROTTLE = 0
        torch.set_num_threads(mp.cpu_count())
        os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
        print(f"[FAST MODE] {NUM_WORKERS} workers, all {mp.cpu_count()} cores, no throttle")
    else:
        NUM_WORKERS = 4
        STEP_THROTTLE = 0.005
    EPISODES_PER_WORKER = 10 if fast_mode else 3

    print(f"[CPU] {NUM_WORKERS} workers × {EPISODES_PER_WORKER} eps/dispatch")

    env = Tetris()
    episodes = 2500000
    max_steps = None
    epsilon_stop_episode = 35000  # Explore for 70% of training
    mem_size = 2000000 if fast_mode else 100000  # 2M entries — 3090 has 24GB VRAM to spare
    discount = 0.95  # Focus on near-term rewards (line clears)
    n_step = 3  # N-step returns — propagates reward 3 steps back per update
    batch_size = 8192 if fast_mode else 2048  # Larger batches saturate GPU compute
    render_every = None
    render_delay = None
    log_every = 50
    replay_start_size = 5000  # More diverse initial buffer
    n_neurons = [2048, 1536, 768, 384]
    activations = ['relu', 'relu', 'relu', 'relu', 'linear']
    lr = 1e-3
    save_best_model = True

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, epsilon_min=0.01,
                     mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size, lr=lr,
                     n_step=n_step)

    # ── Resume from checkpoint ─────────────────────────────────
    scores = []
    best_score = 0
    start_episode = 0

    if resume_from and os.path.exists(resume_from):
        start_episode, best_score = agent.load_checkpoint(resume_from)
        # Add current episode target ON TOP of where we left off
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

    pool = WorkerPool(NUM_WORKERS, step_throttle=STEP_THROTTLE,
                      n_step=n_step, discount=discount)

    # ── Live Dashboard ───────────────────────────────��──────────────
    dashboard_data = {
        'episode': 0, 'total_episodes': episodes,
        'avg_score': 0, 'best_score': 0, 'epsilon': 1.0,
        'memory': 0, 'gpu_passes': 0,
        'gpu_util': 0, 'gpu_peak_util': 0,
        'gpu_temp': 0, 'gpu_peak_temp': 0,
        'gpu_power': 0, 'gpu_clock': 0,
    }

    # ── Persistent GPU training thread ──────────────────────────────
    # Runs for the ENTIRE session. Never stops until shutdown.
    # NO LOCK on the training hot path!
    # Safety: train() reads random indices < _mem_count,
    #         flush_to_gpu() writes to [_mem_pos..+n) then bumps _mem_count.
    #         Training never sees half-written data because _mem_count is
    #         updated AFTER the bulk write completes.
    # weights_lock: brief lock ONLY for save_model/save_checkpoint (GPU→CPU sync).
    weights_lock = threading.Lock()
    gpu_shutdown = threading.Event()
    gpu_passes = [0]

    # GPU governor — fixed params with hard utilization ceiling + thermal monitoring
    # clock_min/clock_max: lock GPU core clock range in MHz (requires admin/root)
    #   Example: clock_min=800, clock_max=1500 for 800-1500 MHz
    #   Set both equal for a perfectly flat clock: clock_min=1200, clock_max=1200
    #   Set to None to leave clocks at default (GPU boosts freely)
    # mem_clock_min/mem_clock_max: lock VRAM clock range in MHz
    #   RTX 4070 VRAM typically: 405-10501 MHz
    #   Example: mem_clock_min=5001, mem_clock_max=5001 for fixed VRAM speed
    if fast_mode:
        # Fast mode: lock GPU clocks HIGH so it never downclocks between bursts
        # RTX 3090: ~1800 MHz core, 9751 MHz memory
        governor = GPUGovernor(target_high=1.0, burst=200, cooldown=0.0,
                               batch_size=batch_size,
                               clock_min=1700, clock_max=1900,
                               mem_clock_min=9501, mem_clock_max=9751)
    else:
        governor = GPUGovernor(target_high=0.70, burst=20, cooldown=0.01,
                               batch_size=batch_size,
                               clock_min=1500, clock_max=1500,
                               mem_clock_min=9001, mem_clock_max=9001)

    # Launch dashboard in daemon thread
    dashboard = TrainingDashboard(dashboard_data, target_high=governor.target_high)
    dash_thread = threading.Thread(target=dashboard.run, daemon=True)
    dash_thread.start()

    def _gpu_train_loop():
        while not gpu_shutdown.is_set():
            n = agent._mem_count
            bs = governor.batch_size
            if n < replay_start_size or n < bs:
                time.sleep(0.1)
                continue

            if not fast_mode:
                # ── HARD CEILING: block until GPU drops below target_high ──
                while not gpu_shutdown.is_set():
                    if not governor.at_ceiling():
                        break
                    time.sleep(0.1)

            # Train a burst of batches — epochs=4 replays each batch 4× to keep
            # the small model busy on the GPU (400K params finishes too fast otherwise)
            for _ in range(governor.burst):
                if gpu_shutdown.is_set():
                    return
                agent.train(batch_size=bs, epochs=4)
                gpu_passes[0] += 1

            if not fast_mode:
                torch.cuda.synchronize()
                time.sleep(governor.cooldown)
            governor.update()

    gpu_thread = threading.Thread(target=_gpu_train_loop, daemon=True)
    gpu_thread.start()

    # ── GPU inference thread ──────────────────────────────────────
    # Workers send (worker_id, states_array) to pool.inference_queue.
    # This thread runs the model forward pass on GPU and returns best_idx.
    inference_shutdown = threading.Event()

    def _gpu_inference_loop():
        device = next(agent.model.parameters()).device
        while not inference_shutdown.is_set():
            try:
                try:
                    request = pool.inference_queue.get(timeout=0.05)
                except Exception:
                    continue

                worker_id, states_array = request

                with torch.no_grad():
                    states_tensor = torch.tensor(states_array, dtype=torch.float32,
                                                  device=device)
                    predictions = agent.model(states_tensor)
                    best_idx = int(torch.argmax(predictions).item())

                pool.response_queues[worker_id].put(best_idx)

            except Exception:
                if inference_shutdown.is_set():
                    break
                continue

    inference_thread = threading.Thread(target=_gpu_inference_loop, daemon=True)
    inference_thread.start()

    try:
        # ── Staggered initial dispatch ─────────────────────────────
        # Spread worker starts over ~200ms so they don't all spike CPU at once
        for i in range(NUM_WORKERS):
            pool.dispatch_one(agent.epsilon, EPISODES_PER_WORKER)
            if i < NUM_WORKERS - 1:
                time.sleep(0.2 / NUM_WORKERS)

        # ── Rolling collection loop ────────────────────────────────
        # Process results one worker at a time. As each finishes,
        # immediately re-dispatch it. Workers are always staggered,
        # so CPU load stays flat — no burst/idle/burst pattern.
        while episode < episodes:
            # Wait for ONE worker to finish
            all_experiences, ep_scores, ep_steps = pool.collect_one()

            # Ingest this worker's experiences
            agent.add_batch_to_memory(all_experiences)
            agent.flush_to_gpu()

            # Immediately re-dispatch this worker (keeps it busy)
            pool.dispatch_one(agent.epsilon, EPISODES_PER_WORKER)

            # Track scores
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

            # ── Render (occasionally) ───────────────────────────────
            if render_every and episode % render_every < round_count:
                current_state = env.reset()
                done = False
                steps = 0
                while not done and (not max_steps or steps < max_steps):
                    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
                    if not next_states:
                        break
                    best_state = agent.best_state(next_states.keys())
                    best_action = next_states[best_state]
                    reward, done = env.play(best_action[0], best_action[1],
                                            render=True, render_delay=render_delay)
                    agent.add_to_memory(current_state, best_state, reward, done)
                    current_state = best_state
                    steps += 1
                agent.flush_to_gpu()
                score = env.get_game_score()
                scores.append(score)
                if save_best_model and score > best_score:
                    best_score = score
                    with weights_lock:
                        agent.save_model("best.pt")

            # ── Dashboard + logging ─────────────────────────────────
            recent_avg = mean(scores[-50:]) if scores else 0
            dashboard_data.update({
                'episode': episode,
                'avg_score': recent_avg,
                'best_score': best_score,
                'epsilon': agent.epsilon,
                'memory': agent._mem_len(),
                'gpu_passes': gpu_passes[0],
                'gpu_util': governor._ema_util,
                'gpu_peak_util': governor._peak_util,
                'gpu_temp': governor._ema_temp,
                'gpu_peak_temp': governor._peak_temp,
                'gpu_power': governor.read_power(),
                'gpu_clock': governor.read_clock(),
            })

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
        inference_shutdown.set()
        gpu_thread.join(timeout=5)
        inference_thread.join(timeout=5)
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
                        help='Fast mode: max CPU, no throttle, no rendering')
    args = parser.parse_args()
    dqn(resume_from=args.resume, fast_mode=args.fast)
