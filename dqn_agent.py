import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import multiprocessing as _mp

# Auto-detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if _mp.current_process().name == 'MainProcess':
    print(f"[DQN] Using device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))


class DQNModel(nn.Module):
    def __init__(self, state_size, n_neurons, activations):
        super().__init__()
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
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    '''Deep Q Learning Agent with GPU-resident replay buffer.

    The entire replay buffer lives on GPU as pre-allocated tensors.
    Training samples directly on GPU with zero CPU involvement,
    keeping the GPU saturated.
    '''

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 lr=1e-3, replay_start_size=None, modelFile=None,
                 n_step=1, per_alpha=0.6, per_beta=0.4):

        if len(activations) != len(n_neurons) + 1:
            raise ValueError(f"Expected activations list of length {len(n_neurons) + 1}")

        self.state_size = state_size
        self.mem_size = mem_size
        self.discount = discount
        self.n_neurons = n_neurons
        self.activations = activations

        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode
        else:
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0

        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size

        # N-step returns
        self.n_step = n_step
        self.discount_n = discount ** n_step

        # Prioritized Experience Replay
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_end = 1.0
        self.per_beta_anneal = 200000
        self.per_epsilon = 1e-6
        self._max_priority = 1.0

        if modelFile is not None:
            self.model = DQNModel(state_size, n_neurons, activations).to(DEVICE)
            self.model.load_state_dict(torch.load(modelFile, map_location=DEVICE, weights_only=True))
        else:
            self.model = DQNModel(state_size, n_neurons, activations).to(DEVICE)

        # Target network — frozen copy for stable Q-value targets
        self.target_model = DQNModel(state_size, n_neurons, activations).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self._train_steps = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Cosine anneal over ~200k train() calls — never fully kills the LR
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200000, eta_min=1e-5
        )
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Element-wise for PER weighting

        # ── GPU-resident replay buffer ──────────────────────────────
        # Pre-allocated on GPU. No CPU→GPU transfer during training.
        self._gpu_states = torch.zeros(mem_size, state_size, device=DEVICE)
        self._gpu_next_states = torch.zeros(mem_size, state_size, device=DEVICE)
        self._gpu_rewards = torch.zeros(mem_size, device=DEVICE)
        self._gpu_dones = torch.zeros(mem_size, dtype=torch.bool, device=DEVICE)
        self._gpu_priorities = torch.ones(mem_size, device=DEVICE)
        self._mem_pos = 0
        self._mem_count = 0  # How many valid entries (up to mem_size)

        # CPU staging buffer for batch inserts
        self._staging_states = []
        self._staging_next_states = []
        self._staging_rewards = []
        self._staging_dones = []

        # Cached weights
        self._cached_weights = None
        self._weights_dirty = True

    def _mem_len(self):
        return self._mem_count

    def add_to_memory(self, current_state, next_state, reward, done):
        '''Stage a single experience (batched flush to GPU later).'''
        self._staging_states.append(current_state)
        self._staging_next_states.append(next_state)
        self._staging_rewards.append(reward)
        self._staging_dones.append(done)

    def add_batch_to_memory(self, experiences):
        '''Stage a batch of experiences.'''
        for state, next_state, reward, done in experiences:
            self._staging_states.append(state)
            self._staging_next_states.append(next_state)
            self._staging_rewards.append(reward)
            self._staging_dones.append(done)

    def flush_to_gpu(self):
        '''Flush all staged experiences to GPU in one bulk transfer.
        Uses at most 2 slice writes to handle circular buffer wraparound.'''
        n = len(self._staging_states)
        if n == 0:
            return

        # Stack into contiguous numpy arrays first, then convert to tensors (fast path)
        states_t = torch.as_tensor(np.array(self._staging_states, dtype=np.float32), device=DEVICE)
        next_t = torch.as_tensor(np.array(self._staging_next_states, dtype=np.float32), device=DEVICE)
        rewards_t = torch.as_tensor(np.array(self._staging_rewards, dtype=np.float32), device=DEVICE)
        dones_t = torch.as_tensor(np.array(self._staging_dones, dtype=np.bool_), device=DEVICE)

        # Bulk write with at most 2 slices for wraparound
        start = self._mem_pos
        if start + n <= self.mem_size:
            # No wraparound — single slice
            self._gpu_states[start:start + n] = states_t
            self._gpu_next_states[start:start + n] = next_t
            self._gpu_rewards[start:start + n] = rewards_t
            self._gpu_dones[start:start + n] = dones_t
            self._gpu_priorities[start:start + n] = self._max_priority
        else:
            # Wraparound — two slices
            first = self.mem_size - start
            self._gpu_states[start:] = states_t[:first]
            self._gpu_next_states[start:] = next_t[:first]
            self._gpu_rewards[start:] = rewards_t[:first]
            self._gpu_dones[start:] = dones_t[:first]
            self._gpu_priorities[start:] = self._max_priority

            remainder = n - first
            self._gpu_states[:remainder] = states_t[first:]
            self._gpu_next_states[:remainder] = next_t[first:]
            self._gpu_rewards[:remainder] = rewards_t[first:]
            self._gpu_dones[:remainder] = dones_t[first:]
            self._gpu_priorities[:remainder] = self._max_priority

        self._mem_pos = (start + n) % self.mem_size
        self._mem_count = min(self._mem_count + n, self.mem_size)

        # Clear staging
        self._staging_states.clear()
        self._staging_next_states.clear()
        self._staging_rewards.clear()
        self._staging_dones.clear()

    @torch.no_grad()
    def best_state(self, states):
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        states = list(states)
        states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        values = self.model(states_t)
        return states[torch.argmax(values).item()]

    def _per_sample(self, batch_size):
        '''Cheap PER via rejection sampling — O(batch_size) with zero buffer scans.
        Oversample uniformly, keep high-priority ones, backfill with uniform.
        Cost: two randint + one index gather + one comparison. Trivial.'''
        n = self._mem_count

        # Oversample 3× uniformly
        oversample = batch_size * 3
        candidates = torch.randint(0, n, (oversample,), device=DEVICE)

        # Accept candidates proportional to their priority
        priorities = self._gpu_priorities[candidates]
        # Acceptance probability = priority / max_priority
        accept_prob = priorities / (self._max_priority + 1e-8)
        rand_vals = torch.rand(oversample, device=DEVICE)
        accepted_mask = rand_vals < accept_prob

        accepted = candidates[accepted_mask]

        if len(accepted) >= batch_size:
            indices = accepted[:batch_size]
        else:
            # Not enough accepted — backfill with uniform random
            shortfall = batch_size - len(accepted)
            backfill = torch.randint(0, n, (shortfall,), device=DEVICE)
            indices = torch.cat([accepted, backfill])

        # IS weights from sampled priorities
        beta = min(1.0, self.per_beta + (self.per_beta_end - self.per_beta)
                   * self._train_steps / self.per_beta_anneal)
        sampled_p = self._gpu_priorities[indices].clamp(min=1e-8)
        is_weights = (1.0 / sampled_p) ** beta
        is_weights = is_weights / is_weights.max()

        return indices, is_weights

    def train(self, batch_size=32, epochs=1):
        '''Train on GPU with rank-based PER + n-step + soft target update.'''
        n = self._mem_count
        if n < self.replay_start_size or n < batch_size:
            return

        # ── PER: rank-based segment sampling (fast) ──
        indices, is_weights = self._per_sample(batch_size)

        # Index directly into GPU tensors — zero copy
        states = self._gpu_states[indices]
        next_states = self._gpu_next_states[indices]
        rewards = self._gpu_rewards[indices]
        dones = self._gpu_dones[indices]

        for _ in range(epochs):
            with torch.no_grad():
                next_qs = self.target_model(next_states).squeeze()

            targets = torch.where(dones, rewards, rewards + self.discount_n * next_qs)
            predictions = self.model(states).squeeze()

            # PER-weighted loss
            element_loss = self.loss_fn(predictions, targets)
            loss = (is_weights * element_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # ── Update priorities ──
        with torch.no_grad():
            td_errors = (predictions - targets).abs() + self.per_epsilon
            self._gpu_priorities[indices] = td_errors
            batch_max = td_errors.max().item()
            # Refresh _max_priority periodically from actual buffer
            # Prevents stale max from killing acceptance rate
            if self._train_steps % 1000 == 0:
                self._max_priority = self._gpu_priorities[:self._mem_count].max().item()
            else:
                self._max_priority = max(self._max_priority * 0.99, batch_max)

        self.scheduler.step()
        self._train_steps += 1

        # Soft (Polyak) target update every 10 steps — amortizes the Python loop cost
        if self._train_steps % 10 == 0:
            tau = 0.05  # 10× larger tau since we update 10× less often (equivalent smoothing)
            for p_target, p_online in zip(self.target_model.parameters(), self.model.parameters()):
                p_target.data.mul_(1.0 - tau).add_(p_online.data * tau)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self._weights_dirty = True

    def save_model(self, name):
        torch.save(self.model.state_dict(), name)

    def save_checkpoint(self, path, episode=0, best_score=0):
        '''Save full training state so training can be resumed.'''
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'train_steps': self._train_steps,
            'mem_pos': self._mem_pos,
            'mem_count': self._mem_count,
            'episode': episode,
            'best_score': best_score,
            'per_max_priority': self._max_priority,
            'n_step': self.n_step,
        }, path)

    def load_checkpoint(self, path):
        '''Restore full training state from a checkpoint. Returns (episode, best_score).'''
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.target_model.load_state_dict(ckpt['target_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.epsilon = ckpt['epsilon']
        self.epsilon_min = ckpt['epsilon_min']
        self.epsilon_decay = ckpt['epsilon_decay']
        self._train_steps = ckpt['train_steps']
        self._mem_pos = ckpt.get('mem_pos', 0)
        self._mem_count = ckpt.get('mem_count', 0)
        self._max_priority = ckpt.get('per_max_priority', 1.0)
        if self._mem_count > 0:
            self._gpu_priorities[:self._mem_count] = self._max_priority
        saved_n = ckpt.get('n_step', 1)
        if saved_n != self.n_step:
            print(f"[WARN] Checkpoint was n_step={saved_n}, now using n_step={self.n_step}")
        self._weights_dirty = True
        return ckpt.get('episode', 0), ckpt.get('best_score', 0)

    def get_weights(self):
        if self._weights_dirty or self._cached_weights is None:
            self._cached_weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
            self._weights_dirty = False
        return self._cached_weights

    def set_weights(self, weights):
        state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)
