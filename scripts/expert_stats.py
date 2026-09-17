"""HW1 Table 1"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize(data_dir):
    rows = []
    for name in ('Ant', 'Humanoid', 'Walker2d', 'Hopper', 'HalfCheetah'):
        env = f'{name}-v2'
        filename = data_dir / f'expert_data_{env}.pkl'
        with filename.open('rb') as handle:
            paths = pickle.load(handle)
        if len(paths) != 2:
            raise ValueError(f'{filename}: expected two expert trajectories, found {len(paths)}')
        returns, lengths = [], []
        for path in paths:
            rewards = np.asarray(path['reward'], dtype=np.float64)
            if rewards.ndim != 1 or not np.isfinite(rewards).all():
                raise ValueError(f'{filename}: invalid rewards')
            length = len(rewards)
            for key in ('observation', 'action', 'next_observation', 'terminal'):
                if len(path[key]) != length:
                    raise ValueError(f'{filename}: inconsistent {key} length')
            returns.append(float(rewards.sum()))
            lengths.append(length)
        mean = float(np.mean(returns))
        rows.append(dict(environment=env, trajectories=len(paths), lengths=lengths,
                         observation_dim=paths[0]['observation'].shape[1],
                         action_dim=paths[0]['action'].shape[1],
                         returns=returns, mean_return=mean,
                         std_return=float(np.std(returns, ddof=0)),
                         thirty_percent_of_expert=.3 * mean))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'hw1/rob831/expert_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/expert_stats')
    args = parser.parse_args()
    rows = summarize(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'table1.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# Table 1: supplied expert demonstrations', '',
             'Undiscounted episode returns; population standard deviation (`ddof=0`), matching the trainer. Sums use float64.', '',
             '| Environment | Lengths | Return 1 | Return 2 | Mean | Std | 30% of mean |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        r1, r2 = row['returns']
        lines.append(f"| {row['environment']} | {row['lengths']} | {r1:.2f} | {r2:.2f} | {row['mean_return']:.2f} | {row['std_return']:.2f} | {row['thirty_percent_of_expert']:.2f} |")
    (args.output_dir / 'table1.md').write_text('\n'.join(lines) + '\n')
    tex = ['% Population std over the two supplied trajectories; undiscounted returns.',
           r'\begin{tabular}{lrr}', r'\hline', r'Environment & Mean return & Standard deviation \\', r'\hline']
    for row in rows:
        tex.append(f"{row['environment']} & {row['mean_return']:.2f} & {row['std_return']:.2f} " + r'\\')
    tex.extend([r'\hline', r'\end{tabular}'])
    (args.output_dir / 'table1.tex').write_text('\n'.join(tex) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved Markdown, JSON, and LaTeX to {args.output_dir}')


if __name__ == '__main__':
    main()
