import sys
import argparse

parser = argparse.ArgumentParser(description="Watch the trained Tetris AI play")
parser.add_argument('model', type=str, help='Model file (e.g. best.pt)')
parser.add_argument('--speed', type=float, default=1.0,
                    help='Playback speed multiplier (default: 1.0, try 2, 5, 10, or 0.5 for slow-mo)')
args = parser.parse_args()

from dqn_agent import DQNAgent
from tetris import Tetris

env = Tetris()
agent = DQNAgent(env.get_state_size(),
                 n_neurons=[64, 64],
                 activations=['relu', 'relu', 'linear'],
                 modelFile=args.model)

# Speed multiplier: 2 = twice as fast, 10 = 10x fast, 0.5 = half speed
# We pass render_delay as inverse of speed — None lets NES gravity control it
if args.speed <= 0:
    args.speed = 1.0

# Calculate delay: at speed=1 use None (NES default), otherwise override
render_delay = None if args.speed == 1.0 else max(0.001, 0.02 / args.speed)

print(f"[REPLAY] Model: {args.model}")
print(f"[REPLAY] Speed: {args.speed}x" + (" (NES default)" if render_delay is None else f" (delay={render_delay*1000:.1f}ms)"))
print(f"[REPLAY] Controls: Close window to exit")
print()

done = False
while not done:
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    if not next_states:
        break
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1],
                            render=True, render_delay=render_delay)

print(f"\nGame Over! Final score: {env.get_game_score()}")
print(f"Lines: {env.total_lines}, Level: {env.level}")
