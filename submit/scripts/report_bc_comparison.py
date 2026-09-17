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
        candidates = sorted((ROOT / 'run_logs').glob(f'q1_table2_bc_{robot}_L3_H64_U1000_seed1_*'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f'No comparison run for {env}; run results/bc_comparison/run.sh first')
        run = candidates[-1]
        events = EventAccumulator(str(run)).Reload()
        metrics = {tag: events.Scalars(tag)[-1].value for tag in events.Tags()['scalars']}
        with (ROOT / f'rob831/expert_data/expert_data_{env}.pkl').open('rb') as handle:
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
                                 for p in sorted((ROOT / 'rob831').rglob('*.py'))})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'results.json').write_text(json.dumps(dict(config=config, results=rows), indent=2) + '\n')
    lines = ['# Table 2: matched behavior-cloning comparison', '',
             '| Environment | Expert mean ± std | BC mean ± std | BC / expert |',
             '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['environment']} | {row['expert_mean']:.2f} ± {row['expert_std']:.2f} | {row['bc_mean']:.2f} ± {row['bc_std']:.2f} | {row['expert_percentage']:.2f}% |")
    (OUTPUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
