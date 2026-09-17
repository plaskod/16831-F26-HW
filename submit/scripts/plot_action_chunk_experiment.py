"""Generate Figure 1: horizon-averaged BC versus its K=1 baseline."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dirs', type=Path, nargs='*', default=[Path('results/action_chunks/main')])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.output_dir or args.result_dirs[0]
    configs = [json.loads((folder / 'config.json').read_text()) for folder in args.result_dirs]
    config = configs[0].copy()
    sources = {}
    for folder, other in zip(args.result_dirs, configs):
        for key in ('horizons', 'seeds', 'updates', 'train_batch_size', 'eval_episodes',
                    'ep_len', 'n_layers', 'size', 'learning_rate', 'optimizer', 'loss',
                    'boundary', 'evaluation_seeds', 'execution', 'initialization'):
            assert other[key] == config[key], f'Mismatched {key}: {folder}'
        for env in other['envs']:
            assert env not in sources, f'Duplicate environment: {env}'
            sources[env] = folder
    config['envs'] = list(sources)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, len(config['envs']), figsize=(5.5 * len(config['envs']), 4.1),
                             squeeze=False, layout='constrained')
    for ax, env in zip(axes[0], config['envs']):
        points = []
        for k in config['horizons']:
            runs = [json.loads((sources[env] / f'{env}_K{k}_seed{s}' / 'result.json').read_text()) for s in config['seeds']]
            returns = np.array([[e['episode_return'] for e in run['episodes']] for run in runs])
            assert returns.shape == (len(config['seeds']), config['eval_episodes'])
            row = dict(environment=env, horizon=k, mean_return=float(returns.mean()),
                       std_return=float(returns.std(ddof=0)), seed_means=returns.mean(axis=1).tolist(),
                       rollout_count=int(returns.size))
            rows.append(row)
            points.append(row)
        baseline = next(r for r in points if r['horizon'] == 1)
        for row in points:
            row['difference_from_bc'] = row['mean_return'] - baseline['mean_return']
        means = [r['mean_return'] for r in points]
        stds = [r['std_return'] for r in points]
        ax.errorbar(config['horizons'], means, yerr=stds, fmt='o-', color='#2563a6',
                    linewidth=1.8, markersize=4, capsize=1.5, elinewidth=.7,
                    capthick=.7, ecolor='#7395bd', zorder=3)
        ax.axhline(baseline['mean_return'], color='#aaaaaa', linestyle=':', linewidth=1)
        ax.annotate('BC', xy=(1, baseline['mean_return']), xycoords=('axes fraction', 'data'),
                    xytext=(5, 0), textcoords='offset points', va='center',
                    color='#888888', fontsize=10, annotation_clip=False)
        ax.set_xscale('log', base=2)
        ax.set_xticks(config['horizons'], [str(k) for k in config['horizons']])
        ax.set(title=env, xlabel='Prediction horizon K', ylabel='Return')
        ax.set_ylim(bottom=min(0, float(np.min(np.array(means) - np.array(stds))) * 1.05))
        ax.grid(axis='y', alpha=.12)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(root / 'figure1.png', dpi=200)
    plt.close(fig)
    (root / 'aggregate.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# Horizon-averaged BC experiment', '',
             '| Environment | K | Mean return | Std | Difference from classic BC |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['environment']} | {row['horizon']} | {row['mean_return']:.2f} | {row['std_return']:.2f} | {row['difference_from_bc']:+.2f} |")
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
