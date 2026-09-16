"""Matched offline-BC sweep: predict K expert actions, execute only the first."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from rob831.agents.bc_agent import BCAgent
from rob831.infrastructure import pytorch_util as ptu

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs', nargs='+', default=['Ant-v2', 'Humanoid-v2'],
                   choices=[n+'-v2' for n in ('Ant','Humanoid','Walker2d','Hopper','HalfCheetah')])
    p.add_argument('--horizons', nargs='+', type=int, default=[1,2,4,8,16])
    p.add_argument('--seeds', nargs='+', type=int, default=[1,2,3])
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--train-batch-size', type=int, default=100)
    p.add_argument('--eval-batch-size', type=int, default=5000)
    p.add_argument('--min-rollouts', type=int, default=5)
    p.add_argument('--ep-len', type=int, default=1000)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=.005)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if min(args.horizons + [args.updates,args.train_batch_size,args.eval_batch_size,args.ep_len,args.min_rollouts]) < 1:
        p.error('Counts and horizons must be positive')
    output = args.output_dir or ROOT/'results/action_chunks'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['output_dir'] = str(output)
    config.update(optimizer='Adam', loss='per-observation mean MSE over valid future offsets',
                  boundary='mask missing targets; no crossing terminal/path boundaries; retain all starts',
                  evaluation_seeds='10000 + 1000 * training_seed + episode_index',
                  execution='replan every step and execute only offset 0',
                  packages={n: importlib.metadata.version(n) for n in ('torch','numpy','gym','mujoco-py')},
                  source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in sorted((ROOT/'hw1/rob831').rglob('*.py'))})
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    all_results = []
    ptu.init_gpu(False)
    for env_name in args.envs:
        dataset = ROOT/f'hw1/rob831/expert_data/expert_data_{env_name}.pkl'
        with dataset.open('rb') as handle:
            paths = pickle.load(handle)
        for horizon in args.horizons:
            for seed in args.seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                env = gym.make(env_name)
                env.reset(seed=seed)
                folder = output/f'{env_name}_K{horizon}_seed{seed}'
                folder.mkdir()
                writer = SummaryWriter(str(folder))
                try:
                    agent = BCAgent(env, dict(ac_dim=env.action_space.shape[0],
                        ob_dim=env.observation_space.shape[0], n_layers=args.n_layers,
                        size=args.size, discrete=False, learning_rate=args.learning_rate,
                        max_replay_buffer_size=1000000, action_chunk_size=horizon))
                    agent.add_to_replay_buffer(paths)
                    losses = []
                    for step in range(args.updates):
                        log = agent.train(*agent.sample(args.train_batch_size))
                        loss = float(log['Training Loss'])
                        if not np.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        losses.append(loss)
                        if step % 50 == 0 or step == args.updates - 1:
                            writer.add_scalar('training/chunk_mse', loss, step)
                    agent.actor.eval()
                    episodes, total_steps = [], 0
                    while total_steps < args.eval_batch_size or len(episodes) < args.min_rollouts:
                        eval_seed = 10000 + 1000 * seed + len(episodes)
                        obs = env.reset(seed=eval_seed)
                        rewards = []
                        for _ in range(args.ep_len):
                            action = agent.actor.get_action(obs)[0]
                            assert action.shape == env.action_space.shape
                            obs, reward, done, info = env.step(action)
                            rewards.append(float(reward))
                            if done:
                                break
                        episodes.append(dict(seed=eval_seed, length=len(rewards),
                                             episode_return=float(np.sum(rewards)), rewards=rewards))
                        total_steps += len(rewards)
                    returns = [e['episode_return'] for e in episodes]
                    result = dict(environment=env_name, horizon=horizon, seed=seed,
                        transitions=len(agent.replay_buffer),
                        parameter_count=sum(t.numel() for t in agent.actor.parameters()),
                        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                        evaluation_steps=total_steps, rollout_count=len(episodes),
                        mean_return=float(np.mean(returns)), std_return=float(np.std(returns)),
                        final_loss=losses[-1], episodes=episodes, training_losses=losses)
                    writer.add_scalar('evaluation/mean_return', result['mean_return'], 0)
                    writer.add_scalar('evaluation/std_return', result['std_return'], 0)
                    for i, value in enumerate(returns):
                        writer.add_scalar('evaluation/episode_return', value, i)
                    agent.save(str(folder/'policy.pt'))
                    (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                    all_results.append(result)
                    print(f"{env_name} K={horizon:2} seed={seed}: {result['mean_return']:.2f} +/- {result['std_return']:.2f}, {len(episodes)} rollouts", flush=True)
                finally:
                    writer.close()
                    env.close()
    # Compact index; detailed trajectories stay in each run folder.
    index = [{k:v for k,v in r.items() if k not in ('episodes','training_losses')} for r in all_results]
    (output/'summary.json').write_text(json.dumps(index, indent=2)+'\n')
    print('Results:', output, flush=True)


if __name__ == '__main__':
    main()
