"""
Quick integration smoke test: import every brain module, build the network,
run one forward pass on CPU with dummy data.
"""
import sys, torch, numpy as np
sys.path.insert(0, '/home/benny/green-vision')

# 1. Network
from src.brain.network import GreenVisionNetwork, SEQ_LEN, ACTION_NAMES, STATE_DIM
net = GreenVisionNetwork()
n_params = sum(p.numel() for p in net.parameters())
print(f'network: {n_params:,} params, actions={len(ACTION_NAMES)}')
frames = torch.randn(1, SEQ_LEN, 3, 64, 64)
state  = torch.randn(1, STATE_DIM)
logits, value = net(frames, state)
assert logits.shape == (1, len(ACTION_NAMES)), f'bad logits {logits.shape}'
assert value.shape == (1,), f'bad value {value.shape}'
print('network forward OK')

# 2. PPO agent (CPU only, tiny config to avoid VRAM)
from src.brain.ppo_agent import PPOAgent, PPOConfig
cfg = PPOConfig(device='cpu', rollout_capacity=8, replay_capacity=16,
                batch_size=4, update_every_steps=4, num_epochs_per_update=1)
agent = PPOAgent(config=cfg)
f = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
s = np.zeros(STATE_DIM, dtype=np.float32)
l, v = agent.observe(f, s)
a, lp = agent.select_action(l)
agent.store_transition(f, s, a, 1.0, False, lp, float(v.item()))
print(f'ppo: action={a} ({ACTION_NAMES[a]}), logp={lp:.3f}, val={v.item():.3f}')

# 3. RSSM world model
from src.brain.rssm_world_model import RSSMWorldModel
wm = RSSMWorldModel(device='cpu')
state_vec = np.random.randn(STATE_DIM).astype(np.float32)
actions = [0, 1, 2]
preds = wm.rollout(state_vec, actions, horizon=3)
print(f'rssm: {len(preds)} rollouts, pred shape {preds[0].shape}')

# 4. Planner
from src.brain.planner import AutonomousPlanner
planner = AutonomousPlanner()
goals = planner.generate_goals_from_state({'hp': 50, 'resources': 10})
print(f'planner: {len(goals)} goals -> {[g.name for g in goals[:3]]}')

# 5. Curiosity
from src.brain.curiosity import CuriosityModule
cur = CuriosityModule()
s1 = np.random.randn(STATE_DIM).astype(np.float32)
s2 = np.random.randn(STATE_DIM).astype(np.float32)
r_intr = cur.compute_intrinsic_reward(s1, 0, s2)
print(f'curiosity: intrinsic={r_intr:.4f}')

print('\nALL MODULES OK')
