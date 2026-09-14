"""Smoke-test dependencies, all supplied experts, rendering, and video logging."""
from pathlib import Path
import argparse
import pickle
import tempfile

import gym
import mujoco
import mujoco_py
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rob831.infrastructure import pytorch_util as ptu
from rob831.infrastructure.logger import Logger
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Also verify OpenGL and TensorBoard video encoding')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / 'hw1'
ptu.init_gpu(use_gpu=False)
print(f'PyTorch {torch.__version__}; Gym {gym.__version__}; native MuJoCo {mujoco.__version__}; legacy binding {mujoco_py.get_version()}', flush=True)
for name in ('Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Humanoid'):
    env_id = f'{name}-v2'
    env = gym.make(env_id)
    try:
        obs = env.reset(seed=0)
        expert = LoadedGaussianPolicy(str(root / 'rob831/policies/experts' / f'{name}.pkl'))
        with (root / 'rob831/expert_data' / f'expert_data_{env_id}.pkl').open('rb') as handle:
            data = pickle.load(handle)
        assert data[0]['observation'].shape[1:] == env.observation_space.shape
        assert data[0]['action'].shape[1:] == env.action_space.shape
        frames = []
        for step in range(10):
            with torch.no_grad():
                action = expert.get_action(obs)[0]
            assert action.shape == env.action_space.shape and np.isfinite(action).all()
            obs, reward, done, info = env.step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if args.render:
                frame = env.sim.render(camera_name='track', height=128, width=128)[::-1]
                assert frame.shape == (128, 128, 3) and frame.std() > 0
                frames.append(frame)
            if done:
                obs = env.reset()
        if args.render:
            with tempfile.TemporaryDirectory(prefix='rob831-check-') as logdir:
                logger = Logger(logdir)
                logger.log_scalar(reward, 'smoke_reward', 0)
                logger.log_paths_as_videos([{'image_obs': np.stack(frames)}], 0, fps=10)
                logger._summ_writer.close()
                events = EventAccumulator(logdir).Reload()
                assert events.Tags()['scalars'] and events.Tags()['images'], events.Tags()
        print(f'PASS {env_id}: obs={env.observation_space.shape}, action={env.action_space.shape}, expert/data/10 steps' + ('/render/video' if args.render else ''), flush=True)
    finally:
        env.close()
# Check CPU autograd and optimizer separately from the incomplete homework MLP.
model = torch.nn.Linear(3, 2)
optimizer = torch.optim.Adam(model.parameters())
loss = model(torch.ones(4, 3)).square().mean()
loss.backward()
optimizer.step()
print('PASS CPU autograd and optimizer', flush=True)
