"""Export Table 2 from the matched runs produced by results/bc_comparison/run.sh."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import pickle

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/bc_comparison'


def main():
    rows = []
    for robot in ('Ant', 'Hopper'):
        env = f'{robot}-v2'
        candidates = sorted((ROOT / 'hw1/data').glob(f'q1_table2_bc_{robot}_L3_H64_U1000_seed1_*'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f'No comparison run for {env}; run results/bc_comparison/run.sh first')
        run = candidates[-1]
        events = EventAccumulator(str(run)).Reload()
        metrics = {tag: events.Scalars(tag)[-1].value for tag in events.Tags()['scalars']}
        with (ROOT / f'hw1/rob831/expert_data/expert_data_{env}.pkl').open('rb') as handle:
            paths = pickle.load(handle)
        expert_returns = [float(np.asarray(p['reward'], dtype=np.float64).sum()) for p in paths]
        expert_mean = float(np.mean(expert_returns))
        rows.append(dict(environment=env, logdir=str(run.relative_to(ROOT)),
                         expert_mean=expert_mean, expert_std=float(np.std(expert_returns)),
                         expert_trajectories=len(paths), expert_transitions=sum(len(p['reward']) for p in paths),
                         bc_mean=metrics['Eval_AverageReturn'], bc_std=metrics['Eval_StdReturn'],
                         expert_percentage=100 * metrics['Eval_AverageReturn'] / expert_mean,
                         evaluation_average_episode_length=metrics['Eval_AverageEpLen'],
                         final_minibatch_mse=metrics['Training_Loss']))
        assert (run / 'policy_itr_0.pt').exists()
    assert rows[0]['expert_percentage'] >= 30 and rows[1]['expert_percentage'] < 30
    config = dict(n_layers=3, size=64, hidden_activation='tanh', output_activation='identity',
                  optimizer='Adam', loss='MSE', learning_rate=.005, n_iter=1,
                  num_agent_train_steps_per_iter=1000, train_batch_size=100,
                  eval_batch_size=5000, ep_len=1000, seed=1, video_log_freq=-1,
                  device='cpu', save_params=True, std_ddof=0,
                  packages={p: importlib.metadata.version(p) for p in ('gym', 'torch', 'numpy', 'mujoco-py')},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / 'hw1/rob831').rglob('*.py'))})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'results.json').write_text(json.dumps(dict(config=config, results=rows), indent=2) + '\n')
    caption = (r'BC vs.\ expert return. Both policies use three hidden layers of 64 tanh units '
               r'(task-specific input/output dimensions), trained on two supplied expert trajectories '
               r'(2,000 transitions) per environment. One BC iteration uses 1,000 Adam updates, '
               r'learning rate 0.005, minibatches of 100, and MSE loss; seed 1, CPU. '
               r'Evaluation uses \texttt{eval\_batch\_size}=5,000 transitions and '
               r'\texttt{ep\_len}=1,000, collecting complete episodes until at least 5,000 steps '
               r'are reached (at least five rollouts; early termination can produce more). '
               r'Mean and population standard deviation are over episode returns, not training seeds. '
               r'Expert statistics use the two supplied demonstrations. Video logging is disabled.')
    a, h = rows
    tex = [r'\subsection{Part 3 (5 pt)}', r'\begin{table}[htbp]', r'  \centering',
           '  \\caption{' + caption + '}', r'  \begin{tabular}{ccccc}', r'    \toprule',
           r'    Env & \multicolumn{2}{c}{Ant-v2} & \multicolumn{2}{c}{Hopper-v2} \\',
           r'    \midrule', r'    Metric & Mean & Std. & Mean & Std. \\',
           f"    Expert & {a['expert_mean']:.2f} & {a['expert_std']:.2f} & {h['expert_mean']:.2f} & {h['expert_std']:.2f} " + r'\\',
           f"    BC & {a['bc_mean']:.2f} & {a['bc_std']:.2f} & {h['bc_mean']:.2f} & {h['bc_std']:.2f} " + r'\\',
           r'    \bottomrule', r'  \end{tabular}', r'  \label{tab:p3}', r'\end{table}']
    (OUTPUT / 'table2.tex').write_text('\n'.join(tex) + '\n')
    lines = ['# Table 2: matched behavior-cloning comparison', '',
             '| Environment | Expert mean ± std | BC mean ± std | BC / expert |',
             '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['environment']} | {row['expert_mean']:.2f} ± {row['expert_std']:.2f} | {row['bc_mean']:.2f} ± {row['bc_std']:.2f} | {row['expert_percentage']:.2f}% |")
    lines.extend(['', 'Both runs meet the requested 30% conditions. Hyperparameters are recorded in results.json and the LaTeX caption.',
                  '', 'Reproduce from the repository root:', '', '```bash',
                  'bash results/bc_comparison/run.sh', '.venv/bin/python scripts/report_bc_comparison.py', '```', '',
                  'These are one-seed results. Standard deviations measure variability across evaluation episodes, not across independently trained policies.',
                  'The same hidden-layer sizes are used; input/output dimensions and therefore total parameter counts differ by task.',
                  'Hopper was selected first: five live expert episodes averaged 3716.14 ± 2.30 versus 3772.67 in saved demonstrations.',
                  'Expert check: results/dagger_comparison/expert_check_Hopper/. Walker2d and HalfCheetah were not needed.',
                  'To rerun only Hopper while preserving Ant: bash results/bc_comparison/run.sh Hopper', '', '## Event logs and checkpoints', ''])
    lines.extend(f"- `{row['logdir']}`" for row in rows)
    (OUTPUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
