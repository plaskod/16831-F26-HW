"""Plot measured chunk-horizon outcomes, keeping episode and seed variability distinct."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args=parser.parse_args()
    root=args.result_dir
    config=json.loads((root/'config.json').read_text())
    rows=[]
    fig, axes=plt.subplots(1,len(config['envs']),figsize=(6*len(config['envs']),4.5),squeeze=False)
    for ax,env in zip(axes[0],config['envs']):
        means,stds=[],[]
        for k in config['horizons']:
            runs=[json.loads((root/f'{env}_K{k}_seed{s}'/'result.json').read_text()) for s in config['seeds']]
            # Equal weight for each trained policy even when episode lengths differ.
            # This is the std of the equal-seed mixture of episode returns,
            # not a standard error or confidence interval.
            seed_means=np.array([r['mean_return'] for r in runs])
            seed_vars=np.array([r['std_return']**2 for r in runs])
            mean=float(seed_means.mean())
            std=float(np.sqrt(np.mean(seed_vars+seed_means**2)-mean**2))
            row=dict(environment=env,horizon=k,mean_return=mean,episode_mixture_std=std,
                     mean_seed_return_std=float(seed_means.std()),seed_means=seed_means.tolist(),
                     rollouts_per_seed=[r['rollout_count'] for r in runs])
            rows.append(row); means.append(mean);stds.append(std)
        ax.errorbar(config['horizons'],means,yerr=stds,fmt='o-',capsize=6,
                    linewidth=2,markersize=7,elinewidth=1.6,color='#2563eb')
        ax.set_xscale('log',base=2)
        ax.set_xticks(config['horizons'],[str(k) for k in config['horizons']])
        ax.set(title=env,xlabel='Prediction horizon K',ylabel='Episode return')
        ax.set_ylim(bottom=min(0, min(np.array(means)-np.array(stds))*1.05))
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Final BC performance after {config['updates']:,} updates: mean ± 1 std", fontsize=13)
    fig.tight_layout()
    fig.savefig(root/'figure1.png',dpi=180)
    fig.savefig(root/'figure1.svg')
    plt.close(fig)
    (root/'aggregate.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Future-action supervision experiment','',
           '| Environment | K | Mean return | Episode std, equal seed weight | Seed means | Rollouts per seed |',
           '|---|---:|---:|---:|---|---|']
    conclusions=[]
    for env in config['envs']:
        group=[r for r in rows if r['environment']==env]
        baseline=next(r for r in group if r['horizon']==1)
        for r in group:
            lines.append(f"| {env} | {r['horizon']} | {r['mean_return']:.2f} | {r['episode_mixture_std']:.2f} | {', '.join(f'{x:.2f}' for x in r['seed_means'])} | {r['rollouts_per_seed']} |")
        better=[r['horizon'] for r in group if 1<r['horizon']<=8 and r['mean_return']>baseline['mean_return']]
        conclusions.append(f"{env}: " + (f"horizon(s) {better} exceeded the K=1 aggregate mean." if better else "none of K=2,4,8 exceeded the K=1 aggregate mean."))
    lines += ['', *conclusions, '',
              f"These are descriptive results from {len(config['seeds'])} training seeds, not a statistical significance test.",
              'Error bars are the population std of episode returns under an equal-weight mixture of seeds:',
              'mean = mean(seed_means); variance = mean(seed_variances + seed_means^2) - mean^2.',
              'This avoids overweighting a seed merely because its failed episodes are shorter.',
              'The figure shows final aggregate means and standard-deviation error bars only; seed means remain in the table.',
              'See aggregate.json for separate standard deviations of the seed means.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    moderate_improvements = [r for r in rows if 1 < r['horizon'] <= 8 and r['mean_return'] >
                             next(b['mean_return'] for b in rows if b['environment']==r['environment'] and b['horizon']==1)]
    interpretation = ('There is descriptive evidence of moderate-horizon benefit in some conditions, but not proof of the proposed mechanism.'
                      if moderate_improvements else
                      'The hypothesized moderate-horizon benefit was not supported under this training budget.')
    outcome=(' '.join(conclusions) + ' ' + interpretation).replace('K=', r'$K$=')
    caption=(r'Future-action prediction as auxiliary supervision for closed-loop BC on Ant-v2 and Humanoid-v2. '
             r'Hypothesis: moderate prediction horizons improve imitation, while large horizons may degrade it. '
             r'We vary only the action-prediction horizon $K\in\{1,2,4,8,16\}$; every step executes only '
             r'the first prediction and replans from the new observation. All conditions use the same two expert '
             f"trajectories (2,000 starting observations), {config['n_layers']} hidden layers of {config['size']} tanh units, Adam with learning rate "
             f"{config['learning_rate']}, {config['updates']} updates, minibatches of {config['train_batch_size']}, and training seeds {config['seeds']}. Missing future targets are masked "
             r'at episode boundaries; the output layer grows with $K$. Each trained policy is evaluated on at least '
             f"{config['min_rollouts']} complete rollouts and {config['eval_batch_size']} steps, with a {config['ep_len']}-step episode limit and matched reset-seed "
             r'schedules across horizons. Blue points and error bars show mean and population standard deviation '
             r'of episode returns with equal weight per training seed after the final training update. '
             + outcome + r' These limited-seed observations do not establish statistical significance or isolate '
             r'representation quality from output-head size and optimization effects.')
    # Relative to the main report in hw1/.
    relative='../results/action_chunks/'+root.name+'/figure1.png'
    tex='\n'.join([r'\begin{figure}[htbp]',r'  \centering',
                   r'  \includegraphics[width=\linewidth]{'+relative+'}',
                   r'  \caption{'+caption+'}',r'  \label{fig:p4}',r'\end{figure}'])
    (root/'figure1.tex').write_text(tex+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
