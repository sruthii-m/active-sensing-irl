"""Build a figure/data packet from saved evaluations; never retrains policies."""
import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path('artifacts/state_uncertainty')
OUT = Path('artifacts/analysis/presentation')
COLORS = ['#2166ac', '#d6604d']


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130,
                         'savefig.dpi': 200, 'axes.titleweight': 'bold'})
    reference_path = Path('artifacts/analysis/alignment_uncertainty.json')
    refs = json.loads(reference_path.read_text())['runs']
    runs, episodes, bins, source_paths = [], [], [], [reference_path]
    for fov in (7, 5, 3):
        for seed in (0, 1):
            p = ROOT / f'partial_obs_fov{fov}_seed{seed}_local'
            a = json.loads((p / 'alignment.json').read_text())
            l = json.loads((p / 'likelihood.json').read_text())
            r = refs[p.name]
            rows = read_csv(p / 'alignment.csv')
            source_paths.extend([p/'alignment.json', p/'alignment.csv', p/'likelihood.json'])
            u = a['unresolved']
            starts = [x for x in rows if int(x['timestep']) == 0]
            run = dict(run=p.name, fov=fov, seed=seed, rollout_success=a['success_rate'],
                       accuracy=l['expert_top1_accuracy'], nll=l['negative_log_likelihood'],
                       expert_log_likelihood=l['paired']['expert_mean_log_likelihood'],
                       likelihood_gap=l['paired']['mean_likelihood_gap'],
                       kl=l['paired']['mean_kl_expert_to_iq'],
                       likelihood_steps=l['steps'],
                       strict=u['strict_agreement_unique_ig'],
                       conditional=u['navigation_conditional_agreement_unique_ig'],
                       navigation_spearman=u['navigation_conditional_mean_spearman'],
                       outside_navigation=u['actual_outside_navigation_fraction'],
                       unresolved_steps=u['steps'], eligible_steps=u['unique_ig_steps'],
                       eligible_episodes=r['eligible_episodes'],
                       initial_entropy_mean=np.mean([float(x['entropy_nats']) for x in starts]),
                       alignment_episodes=len(starts),
                       reference=r['action_frequency_reference'], excess=r['excess_agreement'])
            for group in ('success', 'failure'):
                g = l['paired_by_outcome'][group]
                run[group+'_accuracy'] = g['greedy_policy_agreement']
                run[group+'_episodes'] = g['episodes']
                run[group+'_steps'] = g['steps']
            for field, refkey in [('strict','strict_agreement'), ('reference','action_frequency_reference'), ('excess','excess_agreement')]:
                run[field+'_ci_low'], run[field+'_ci_high'] = r[refkey+'_ci95']
            assert abs(run['strict'] - r['strict_agreement']) < 1e-10
            for ep in sorted({int(x['episode']) for x in rows}):
                e = [x for x in rows if int(x['episode']) == ep]
                unresolved = [x for x in e if x['resolved'] == 'False']
                eligible = [x for x in unresolved if x['ig_tie'] == 'False']
                episodes.append(dict(run=p.name, fov=fov, training_seed=seed, episode=ep,
                                     reset_seed=int(e[0]['seed']), length=len(e),
                                     initial_entropy=float(e[0]['entropy_nats']),
                                     unresolved_steps=len(unresolved), eligible_steps=len(eligible),
                                     strict_matches=sum(x['strict_agreement']=='True' for x in eligible)))
            for label, b in r['bins'].items():
                bins.append(dict(run=p.name, fov=fov, seed=seed, candidate_bin=label,
                                 steps=b['steps'], episodes=b['episodes'], reference=b['reference']))
            runs.append(run)
    write_csv('run_summary.csv', runs)
    write_csv('alignment_episodes.csv', episodes)
    write_csv('reference_bins.csv', bins)
    (OUT/'sources.json').write_text(json.dumps([str(p.resolve()) for p in source_paths], indent=2))
    labels = [f"FOV {r['fov']} / seed {r['seed']}" for r in runs]
    x = np.arange(len(runs))
    colors = [COLORS[r['seed']] for r in runs]
    captions = []
    pdf = PdfPages(OUT/'results.pdf')

    def save(fig, name, caption):
        fig.savefig(OUT/(name+'.png'), bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        captions.append(name+'\n'+caption+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0,0]
    ax.bar(x, [r['rollout_success']*100 for r in runs], color=colors)
    ax.set(title='Task completion on 100 alignment episodes', ylabel='Success (%)', ylim=(0,110))
    for i,r in enumerate(runs): ax.text(i, r['rollout_success']*100+1, f"{r['rollout_success']:.0%}", ha='center')
    ax = axes[0,1]
    for delta, key, label, color in [(-.23,'accuracy','All expert steps','#444444'), (0,'success_accuracy','Successful expert episodes','#2a9d8f'), (.23,'failure_accuracy','Failed expert episodes','#e9a03b')]:
        ax.bar(x+delta, [r[key]*100 if r[key] is not None else np.nan for r in runs], width=.23, label=label, color=color)
    ax.set(title='Expert reproduction on held-out histories', ylabel='Action agreement (%)', ylim=(0,110))
    ax.legend(fontsize=8, loc='lower left')
    axes[1,0].bar(x, [r['nll'] for r in runs], color=colors)
    axes[1,0].set(title='IQ negative log likelihood of expert actions', ylabel='Nats / step (lower is better)')
    axes[1,1].bar(x, [r['kl'] for r in runs], color=colors)
    axes[1,1].set(title='Full stochastic policy distribution mismatch', ylabel='KL(expert || IQ), nats / step')
    for ax in axes.flat: ax.set_xticks(x, labels, rotation=28, ha='right')
    fig.suptitle('1. Task success and expert reproduction answer different questions', fontsize=15)
    save(fig,'01_success_and_fidelity', 'Task success uses IQ on-policy alignment rollouts. Fidelity uses separate held-out expert histories; each metric is step-weighted. Missing failure bars mean no failed expert episodes, not zero agreement. Training used successful, length-filtered demos. Two fixed checkpoints per FOV; no training-seed intervals. See success/failure episode and step counts in run_summary.csv.')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, key, title, ylabel in [(axes[0],'initial_entropy_mean','Uncertainty at reset','Mean entropy (nats)'), (axes[1],'unresolved_steps','Time spent searching','Mean unresolved steps / episode'), (axes[2],'eligible_steps','Evidence for strict agreement','Unique-IG unresolved steps')]:
        values=[r[key]/r['alignment_episodes'] if key=='unresolved_steps' else r[key] for r in runs]
        ax.bar(x,values,color=colors)
        ax.set(title=title,ylabel=ylabel)
        ax.set_xticks(x,labels,rotation=40,ha='right')
    fig.suptitle('2. Narrower observations increase initial uncertainty and prolong search',fontsize=15)
    save(fig,'02_uncertainty_and_search','Same 100 reset seeds across FOVs. Initial entropy is measured after the reset observation; search duration depends on the policy. Eligible steps exclude IG ties and resolved states. Steps are not independent experimental replicates. Per-episode values are in alignment_episodes.csv.')

    fig, axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].bar(x-.17,[r['strict']*100 for r in runs],width=.34,color='#2166ac',label='Actual seven-action policy')
    axes[0].bar(x+.17,[r['conditional']*100 for r in runs],width=.34,color='#91bfdb',label='Navigation-only diagnostic')
    axes[0].set(title='Agreement with unique highest-IG action',ylabel='Eligible-step agreement (%)',ylim=(0,100))
    axes[0].legend(fontsize=9)
    axes[1].bar(x,[r['outside_navigation']*100 for r in runs],color=colors)
    axes[1].set(title='Non-navigation choices during unresolved states',ylabel='All unresolved steps (%)',ylim=(0,25))
    for ax in axes: ax.set_xticks(x,labels,rotation=30,ha='right')
    fig.suptitle('3. Strict agreement measures executed behavior',fontsize=15)
    save(fig,'03_strict_vs_conditional','Primary agreement uses actual argmax over seven actions, with pickup/drop/toggle/done counted as disagreement. Conditional scores restrict Q to navigation on the same eligible steps. The right panel uses all unresolved steps, including IG ties, so its denominator differs from the left panel.')

    fig, axes=plt.subplots(1,2,figsize=(12,5.3),constrained_layout=True)
    y=np.arange(len(runs))
    for key, offset, color, label in [('strict',-.12,'#2166ac','Observed strict agreement'),('reference',.12,'#888888','Action-frequency reference')]:
        vals=np.array([r[key]*100 for r in runs])
        lo=np.array([r[key+'_ci_low']*100 for r in runs]); hi=np.array([r[key+'_ci_high']*100 for r in runs])
        axes[0].errorbar(vals,y+offset,xerr=[vals-lo,hi-vals],fmt='o',color=color,capsize=3,label=label)
    vals=np.array([r['excess']*100 for r in runs])
    lo=np.array([r['excess_ci_low']*100 for r in runs]); hi=np.array([r['excess_ci_high']*100 for r in runs])
    axes[1].errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt='o',color='#2a9d8f',capsize=4)
    axes[1].axvline(0,color='black',ls='--',lw=1)
    for ax in axes: ax.set_yticks(y,labels);ax.invert_yaxis()
    axes[0].set(xlabel='Agreement (%)',title='Observed vs frequency-preserving reference')
    axes[0].legend(fontsize=8,loc='lower right')
    axes[1].set(xlabel='Excess agreement (percentage points)',title='Difference with 95% episode-bootstrap interval')
    fig.suptitle('4. All six runs exceed the specified action-frequency reference',fontsize=15)
    save(fig,'04_frequency_reference','Reference preserves all seven action frequencies within candidate-count bins 2-3, 4-7, 8-15, 16-31, 32-63, 64+. Intervals: 5000 whole-episode bootstrap draws, recomputing the reference each draw. Exploratory fixed-policy intervals, not training-seed uncertainty or causal evidence. Geometry/history confounds remain; do not infer a FOV contrast from these intervals alone.')

    p=ROOT/'partial_obs_fov3_seed0_local'
    rows=read_csv(p/'alignment.csv')
    epdata=[e for e in episodes if e['run']==p.name and e['unresolved_steps']>0 and e['unresolved_steps']<e['length']]
    selected=sorted(epdata,key=lambda e:(e['unresolved_steps'],e['episode']))[len(epdata)//2]
    e=[r for r in rows if int(r['episode'])==selected['episode']]
    t=np.array([int(r['timestep']) for r in e]); entropy=np.array([float(r['entropy_nats']) for r in e])
    boundary=next(int(r['timestep']) for r in e if r['resolved']=='True')
    fig, axes=plt.subplots(3,1,figsize=(11,7),sharex=True,constrained_layout=True)
    axes[0].step(t,entropy,where='post',color='#2166ac'); axes[0].set(ylabel='Belief entropy (nats)')
    for a,label in enumerate(['left','right','forward']): axes[1].plot(t,[float(r[f'ig_{a}']) for r in e],label=label)
    axes[1].set(ylabel='One-step IG (nats)'); axes[1].legend(ncol=3,fontsize=9)
    axes[2].scatter(t,[int(r['executed_action']) for r in e],s=28,color='#444444',label='Executed action')
    eligible=[r for r in e if r['resolved']=='False' and r['ig_tie']=='False']
    axes[2].scatter([int(r['timestep']) for r in eligible],[int(r['ig_argmax']) for r in eligible],marker='x',s=50,color='#d6604d',label='Unique IG argmax (unresolved only)')
    axes[2].set(yticks=range(7),yticklabels=['left','right','forward','pickup','drop','toggle','done'],xlabel='Decision timestep')
    axes[2].legend(fontsize=8,loc='upper right')
    for ax in axes:
        ax.axvline(boundary,color='#2a9d8f',ls='--')
        ax.axvspan(boundary,t[-1]+.5,color='#2a9d8f',alpha=.08)
    fig.suptitle(f"5. Entropy resolves; the policy continues acting (FOV 3, seed 0, episode {selected['episode']})",fontsize=13)
    save(fig,'05_episode_example',f"Illustrative, not a statistical aggregate. Selected episode {selected['episode']} by median unresolved duration among FOV3/seed0 episodes with both phases; reset seed {selected['reset_seed']}. Dashed line marks first resolved pre-action belief. IG becomes zero; post-resolution action agreement is not meaningful. No IG argmax markers are drawn for tied states. Raw selected trajectory: example_trajectory.csv.")
    write_csv('example_trajectory.csv',e)

    old=Path('artifacts/analysis')
    legacy=[]
    for name,label in [('existing_demo_likelihood','Original checkpoint / original demos'),('paired_likelihood','Original checkpoint / fresh unfiltered'),('paired_likelihood_1600demos','1600-demo checkpoint / same fresh set')]:
        source_paths.append(old/(name+'.json'))
        d=json.loads((old/(name+'.json')).read_text());legacy.append(dict(comparison=label,accuracy=d['expert_top1_accuracy'],nll=d['negative_log_likelihood'],steps=d['steps'],source=str(old/(name+'.json'))))
    write_csv('legacy_fidelity.csv',legacy)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,key,title in [(axes[0],'accuracy','Expert-action agreement (%)'),(axes[1],'nll','Negative log likelihood (nats/step)')]:
        vals=[r[key]*(100 if key=='accuracy' else 1) for r in legacy]
        ax.bar(range(3),vals,color=['#888888','#d6604d','#2166ac'])
        ax.set_xticks(range(3),['Original model\noriginal demos','Original model\nfresh demos','1600-demo model\nsame fresh demos'])
        ax.set_title(title)
    fig.suptitle('6. Earlier distractor-condition checkpoints exposed an imitation gap',fontsize=14)
    save(fig,'06_legacy_generalization','Historical distractor-condition analysis; do not mix with zero-distractor FOV comparisons. Original-demo membership in training is not verified by legacy metadata. Both fresh-data columns use the same unfiltered trajectories. Architecture/training settings differ between checkpoints, so this is not an isolated demonstration-count ablation.')
    reward_source = ROOT/'partial_obs_fov3_seed1_local'/'reward.png'
    reward_data = reward_source.with_suffix('.npz')
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    ax.imshow(plt.imread(reward_source)); ax.axis('off')
    fig.suptitle('7. Reward inspection: FOV 3, seed 1, one sampled layout', fontsize=14)
    save(fig, '07_reward_example', 'Existing reward map for reset seed 10000, FOV3/seed1. Star marks goal, occupancy shows sampling, blank cells are unsampled. Action-specific recovered rewards differ; this does not show unique reward recovery or a proven goal-centered reward peak. Values average over visited histories/directions and mask terminal bootstrapping. Data are in reward_example.npz; samples columns are x,y,direction followed by seven action rewards.')
    shutil.copyfile(reward_data, OUT/'reward_example.npz')
    source_paths.extend([reward_source, reward_data])

    fig, axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].bar(x-.17, [r['expert_log_likelihood'] for r in runs], width=.34, color='#2a9d8f', label='Expert')
    axes[0].bar(x+.17, [-r['nll'] for r in runs], width=.34, color='#2166ac', label='IQ')
    axes[0].set(title='Probability assigned to recorded expert actions', ylabel='Mean log likelihood (nats / step; closer to 0 is higher)')
    axes[0].legend()
    axes[1].bar(x, [r['likelihood_gap'] for r in runs], color=colors)
    axes[1].axhline(0,color='black',lw=1)
    axes[1].set(title='Expert-minus-IQ log likelihood', ylabel='Mean gap (nats / step)')
    for ax in axes: ax.set_xticks(x, labels, rotation=30, ha='right')
    fig.suptitle('8. Paired likelihood compares both policies on identical histories', fontsize=14)
    save(fig, '08_paired_likelihood', 'Teacher-forced expert histories, recurrence reset each episode, step-weighted. IQ temperature .001. A positive gap means the saved expert assigns more probability to its demonstrated actions on average. Deterministic expert actions were recorded: this gap is not a KL estimate, can be negative, and a zero gap does not guarantee equal distributions. Use figure 1 KL and action agreement alongside it.')
    pdf.close()
    (OUT/'sources.json').write_text(json.dumps([str(p.resolve()) for p in source_paths], indent=2))
    (OUT/'captions.txt').write_text('\n'.join(captions))
    (OUT/'data_dictionary.txt').write_text(
        'All proportions in run_summary.csv are fractions, not percentages.\n'
        'strict and conditional: unresolved steps with unique IG maximizer.\n'
        'outside_navigation: all unresolved steps, including IG ties.\n'
        'accuracy/nll/kl/log-likelihood: held-out expert histories, step-weighted.\n'
        'success/failure prefixes refer to expert trajectory outcomes.\n'
        'rollout_success: IQ on-policy alignment episode success fraction.\n'
        'initial_entropy_mean: nats after observing the reset observation.\n'
        'reference: expected agreement preserving seven-action frequencies within candidate bins.\n'
        'excess: strict minus reference; multiply by 100 for percentage points.\n'
        '*_ci_low/high: 95% percentile whole-episode bootstrap, fixed trained policy.\n'
        'Blank cells in CSVs mean undefined, never zero.\n'
        'alignment_episodes.csv: per-episode data; length/unresolved/eligible are decision-step counts.\n'
        'legacy_fidelity.csv: separate historical distractor condition; checkpoint differences confounded.\n'
        'See captions.txt for selection rules and inferential limitations.\n')
    with zipfile.ZipFile(OUT/'figures_and_data.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.name)
    print(f'Saved eight figures, PDF, data tables, captions, and figures_and_data.zip under {OUT}')


if __name__=='__main__':
    main()
