"""Generate Figure 2 from results/dagger_comparison/run.sh and Table 2."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/dagger_comparison'


def main():
    baseline = json.loads((ROOT / 'results/bc_comparison/results.json').read_text())
    rows = []
    for reference in baseline['results']:
        env = reference['environment']
        robot = env.split('-')[0]
        candidates = sorted((ROOT / 'run_logs').glob(
            f'q2_figure2_dagger_{robot}_L3_H64_U1000_seed1_*'),
            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f'Run results/dagger_comparison/run.sh first: missing {env}')
        run = candidates[-1]
        events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
        tags = ['Eval_AverageReturn', 'Eval_StdReturn', 'Eval_AverageEpLen',
                'Train_EnvstepsSoFar', 'Training_Loss']
        metrics = {}
        for tag in tags:
            values = events.Scalars(tag)
            assert [v.step for v in values] == list(range(10)), (run, tag)
            metrics[tag] = [v.value for v in values]
            assert np.isfinite(metrics[tag]).all(), (run, tag)
        assert np.isclose(metrics['Eval_AverageReturn'][0], reference['bc_mean'], atol=1e-4, rtol=0)
        assert np.isclose(metrics['Eval_StdReturn'][0], reference['bc_std'], atol=1e-4, rtol=0)
        steps = metrics['Train_EnvstepsSoFar']
        assert steps[0] == 0 and np.all(np.diff(steps) >= 1000)
        assert all((run / f'policy_itr_{i}.pt').exists() for i in range(10))
        rows.append(dict(environment=env, logdir=str(run.relative_to(ROOT)),
                         expert_mean=reference['expert_mean'], bc_mean=reference['bc_mean'],
                         bc_std=reference['bc_std'], iterations=list(range(10)), metrics=metrics,
                         final_training_transitions=2000 + int(steps[-1])))

    config = dict(baseline['config'])
    config.update(n_iter=10, do_dagger=True, action_chunk_size=1, batch_size=1000,
                  initial_expert_transitions=2000, max_replay_buffer_size=1000000,
                  scalar_log_freq=1,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / 'rob831').rglob('*.py'))})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'results.json').write_text(json.dumps(dict(config=config, results=rows), indent=2) + '\n')

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for ax, row in zip(axes, rows):
        m = row['metrics']
        ax.errorbar(row['iterations'], m['Eval_AverageReturn'], yerr=m['Eval_StdReturn'],
                    color='#2563a6', marker='o', markersize=4, linewidth=1.8,
                    elinewidth=.7, capsize=1.5, capthick=.7,
                    ecolor='#7395bd', label='DAgger', zorder=3)
        ax.axhline(row['expert_mean'], color='#666666', linestyle='--', linewidth=.9,
                   zorder=2)
        ax.axhline(row['bc_mean'], color='#aaaaaa', linestyle=':', linewidth=1,
                   zorder=2)
        # Offset the close Ant references slightly so their labels remain distinct.
        for name, value, color, offset in [
                ('Expert', row['expert_mean'], '#666666', 5),
                ('BC', row['bc_mean'], '#888888', -5)]:
            ax.annotate(name, xy=(1, value), xycoords=('axes fraction', 'data'),
                        xytext=(5, offset), textcoords='offset points',
                        va='center', fontsize=10, color=color, annotation_clip=False)
        ax.set(title=row['environment'], xlabel='DAgger iteration',
               ylabel='Return', xticks=range(10), xlim=(-.25, 9.25))
        ax.grid(axis='y', alpha=.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    fig.savefig(OUTPUT / 'figure2.png', dpi=200)
    plt.close(fig)

    lines = ['# Figure 2: DAgger', '',
             '| Environment | BC mean ± std | Final DAgger mean ± std | Expert mean | Final training transitions |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        lines.append(f"| {row['environment']} | {row['bc_mean']:.2f} ± {row['bc_std']:.2f} | "
                     f"{m['Eval_AverageReturn'][-1]:.2f} ± {m['Eval_StdReturn'][-1]:.2f} | "
                     f"{row['expert_mean']:.2f} | {row['final_training_transitions']} |")
    (OUTPUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
