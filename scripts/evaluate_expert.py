"""Inspect saved expert reward traces or evaluate experts in fresh Gym rollouts.

This standalone tool never trains a policy or modifies the supplied datasets.
"""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('Ant', 'Humanoid', 'Walker2d', 'Hopper', 'HalfCheetah')


def live_episodes(name, episodes, max_steps, seed):
    import gym
    import torch
    from rob831.infrastructure import pytorch_util as ptu
    from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy
    ptu.init_gpu(use_gpu=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = LoadedGaussianPolicy(str(ROOT / 'hw1/rob831/policies/experts' / f'{name}.pkl'))
    policy.eval()
    env = gym.make(f'{name}-v2')
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs = env.reset(seed=episode_seed)
            rewards, infos = [], []
            done = False
            for _ in range(max_steps):
                with torch.no_grad():
                    action = policy.get_action(obs)[0]
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                infos.append({key: float(value) for key, value in info.items()
                              if isinstance(value, (int, float, np.number))})
                if done:
                    break
            yield dict(episode=episode, seed=episode_seed, rewards=rewards,
                       environment_done=bool(done),
                       time_limit_truncated=bool(info.get('TimeLimit.truncated', False)),
                       evaluator_cutoff=bool(not done and len(rewards) == max_steps),
                       info=infos)
    finally:
        env.close()


def saved_episodes(name):
    filename = ROOT / 'hw1/rob831/expert_data' / f'expert_data_{name}-v2.pkl'
    with filename.open('rb') as handle:
        paths = pickle.load(handle)
    for i, path in enumerate(paths):
        yield dict(episode=i, seed=None, rewards=np.asarray(path['reward'], dtype=float).tolist())


def save_results(folder, name, episodes):
    folder.mkdir()
    for episode in episodes:
        rewards = np.asarray(episode['rewards'], dtype=np.float64)
        if rewards.size == 0 or not np.isfinite(rewards).all():
            raise ValueError('Expected nonempty finite reward sequence')
        episode['length'] = len(rewards)
        episode['return'] = float(rewards.sum())
        episode['cumulative_return'] = rewards.cumsum().tolist()
    returns = [e['return'] for e in episodes]
    summary = dict(environment=f'{name}-v2', episodes=len(episodes),
                   lengths=[e['length'] for e in episodes], returns=returns,
                   mean_return=float(np.mean(returns)),
                   population_std=float(np.std(returns, ddof=0)),
                   sample_std=float(np.std(returns, ddof=1)) if len(returns) > 1 else None)
    (folder / 'steps.json').write_text(json.dumps(episodes, indent=2) + '\n')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for episode in episodes:
        t = np.arange(1, episode['length'] + 1)
        label = f"Episode {episode['episode'] + 1} (R={episode['return']:.1f})"
        axes[0].plot(t, episode['rewards'], alpha=.65, linewidth=1)
        axes[1].plot(t, episode['cumulative_return'], alpha=.8, label=label)
    axes[0].set(title=f'{name}-v2: expert reward traces', ylabel='Reward per step')
    axes[1].set(xlabel='Environment step (1-based)', ylabel='Cumulative return')
    if len(episodes) <= 10:
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'reward_traces.png', dpi=160)
    plt.close(fig)
    print(f"{name}-v2: {summary['mean_return']:.2f} +/- {summary['population_std']:.2f}, n={len(episodes)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('saved', 'live'), default='saved')
    parser.add_argument('--env', choices=('all',) + NAMES, default='all')
    parser.add_argument('--episodes', type=int, default=20, help='Fresh episodes per environment in live mode only')
    parser.add_argument('--max-steps', type=int, default=1000, help='Episode limit in live mode only')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, help='Must not already exist')
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error('--episodes and --max-steps must be positive')
    output = args.output_dir or ROOT / 'results/expert_evaluations' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.source)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(source=args.source, environments=list(NAMES) if args.env == 'all' else [args.env],
                  episodes_per_environment=args.episodes if args.source == 'live' else 'all supplied (2)',
                  max_steps=args.max_steps if args.source == 'live' else None,
                  seed=args.seed if args.source == 'live' else None,
                  policy='Supplied deterministic get_action; no training',
                  packages={p: importlib.metadata.version(p) for p in ('numpy', 'gym', 'torch', 'mujoco-py')})
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print('Saving results to:', output, flush=True)
    summaries = []
    for name in config['environments']:
        episodes = list(saved_episodes(name) if args.source == 'saved' else
                        live_episodes(name, args.episodes, args.max_steps, args.seed))
        summaries.append(save_results(output / f'{name}-v2', name, episodes))
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
