"""Development-only decoder-guided memory edits with norm-matched controls.

Optimizes only a frozen belief decoder, never policy logits. This diagnostic can
fool a decoder or perturb other latent information; it is not causal proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from analysis.policy_io import load_policy


def edit_memory(decoder, x, target, hidden_dim, radius, steps):
    delta = torch.zeros(hidden_dim, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=.02)
    best_loss, best = float('inf'), delta.detach().clone()
    def evaluate(d):
        edited = torch.cat((x[:hidden_dim]+d,x[hidden_dim:]))
        return -(target*decoder(edited).log_softmax(-1)).sum()
    with torch.no_grad():
        best_loss = float(evaluate(delta))
    for _ in range(steps):
        loss = evaluate(delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.mul_(min(1., radius / max(float(delta.norm()),1e-12)))
            value = float(evaluate(delta))
            if value < best_loss:
                best_loss, best = value, delta.clone()
    return best.detach()


def control_deltas(decoder, x, hidden_dim, norm, count, generator):
    def logits(hidden):
        value = decoder(torch.cat((hidden,x[hidden_dim:])))
        return value-value.mean()
    jacobian = torch.autograd.functional.jacobian(logits,x[:hidden_dim]).detach()
    _, singular, vt = torch.linalg.svd(jacobian,full_matrices=False)
    rank = int((singular > max(float(singular.max())*1e-6,1e-8)).sum())
    basis=vt[:rank]
    random=torch.randn(count,hidden_dim,generator=generator)
    null=random-(random@basis.T)@basis
    def normalize(values):
        return values / values.norm(dim=1,keepdim=True).clamp_min(1e-12)*norm
    return normalize(random),normalize(null),rank


def paired_summary(rows,key):
    pairs=sorted({r['pair_id'] for r in rows})
    if not pairs: return dict(mean=None,ci95=None,pairs=0)
    values=np.array([np.mean([r[key] for r in rows if r['pair_id']==p]) for p in pairs])
    draws=np.random.default_rng(0).integers(len(values),size=(2000,len(values)))
    return dict(mean=float(values.mean()),pairs=len(values),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist() if len(values)>1 else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-run',required=True)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--radius-rms',type=float,default=.1)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--controls',type=int,default=20)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if not 0 < args.radius_rms <= 1 or args.steps<1 or args.controls<1:
        parser.error('Need positive edit budget, steps, and control count; radius RMS <=1.')
    out=Path(args.output_dir)
    if out.exists(): parser.error('Choose a new output directory.')
    source=Path(args.natural_run)
    summary=json.loads((source/'summary.json').read_text())
    provenance=json.loads(Path(args.probe).with_name('summary.json').read_text())
    if summary['checkpoint_sha256'] != provenance['checkpoint_sha256']:
        raise ValueError('Probe and natural histories must use the same policy.')
    config=SimpleNamespace(**summary['config'])
    if hashlib.sha256(Path(config.checkpoint).read_bytes()).hexdigest()!=summary['checkpoint_sha256']:
        raise ValueError('Policy checkpoint has changed since rollout collection.')
    with np.load(source/'rollouts.npz',allow_pickle=False) as saved:
        data={k:saved[k] for k in saved.files}
    pairs=json.loads((source/'intervention_candidates.json').read_text())
    state=torch.load(args.probe,map_location='cpu',weights_only=True)
    decoder=nn.Sequential(nn.Linear(state['input_dim'],state['width']),nn.ReLU(),
                          nn.Linear(state['width'],state['output_dim']))
    decoder.load_state_dict(state['state_dict'])
    decoder.eval().requires_grad_(False)
    agent=load_policy(config,data['observation'].shape[1])
    agent.q_net.requires_grad_(False)
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(args.seed)
    dim=data['hidden'].shape[1]
    mean,scale=state['mean'],state['scale']
    radius=args.radius_rms*np.sqrt(dim)
    rows,edits=[],{}
    for pair in pairs:
        for source_side,target_side in [('a','b'),('b','a')]:
            i,j=pair['row_'+source_side],pair['row_'+target_side]
            if data['split'][i]==2 or data['split'][j]==2:
                raise ValueError('Reserved evaluation episodes cannot be used for development edits.')
            hidden=torch.tensor(data['hidden'][i],dtype=torch.float32)
            features=torch.cat((hidden,torch.tensor(data['pose'][i],dtype=torch.float32)))
            x=(features-mean)/scale
            target=torch.tensor(data['belief'][j],dtype=torch.float32)
            target_action=pair['stable_action_'+target_side]
            source_action=pair['stable_action_'+source_side]
            delta=edit_memory(decoder,x,target,dim,radius,args.steps)
            random,null,rank=control_deltas(decoder,x,dim,float(delta.norm()),args.controls,generator)
            def evaluate(changes):
                z=x.repeat(len(changes),1)
                z[:,:dim]+=changes
                h=z[:,:dim]*scale[:dim]+mean[:dim]
                with torch.no_grad():
                    p=(agent.q_net(h)/config.init_temp).softmax(-1)
                    ce=-(target*decoder(z).log_softmax(-1)).sum(-1)
                return (p[:,target_action]-p[:,source_action]).numpy(),ce.numpy(),h.numpy()
            original_margin,original_ce,_=evaluate(torch.zeros(1,dim))
            targeted_margin,targeted_ce,edited=evaluate(delta[None])
            random_margin,random_ce,_=evaluate(random)
            null_margin,null_ce,_=evaluate(null)
            # Re-evaluating the unmodified state must reproduce the original policy.
            restored_margin,_,_=evaluate(torch.zeros(1,dim))
            np.testing.assert_array_equal(original_margin,restored_margin)
            shift=float(targeted_margin[0]-original_margin[0])
            random_shift=float(random_margin.mean()-original_margin[0])
            null_shift=float(null_margin.mean()-original_margin[0])
            row=dict(pair_id=pair['pair_id'],direction=source_side+'->'+target_side,
                     row_source=i,row_target=j,source_action=source_action,target_action=target_action,
                     targeted_shift=shift,random_shift=random_shift,null_shift=null_shift,
                     excess_over_random=shift-random_shift,excess_over_null=shift-null_shift,
                     target_decoder_ce_improvement=float(original_ce[0]-targeted_ce[0]),
                     random_decoder_ce_improvement=float(original_ce[0]-random_ce.mean()),
                     null_decoder_ce_improvement=float(original_ce[0]-null_ce.mean()),
                     standardized_edit_norm=float(delta.norm()),decoder_jacobian_rank=rank,
                     edited_hidden_outside_gru_bounds=bool(np.any(np.abs(edited)>1)))
            rows.append(row)
            prefix=f"pair{pair['pair_id']}_{source_side}_"
            edits[prefix+'hidden_before']=hidden.numpy()
            edits[prefix+'hidden_after']=edited[0]
            edits[prefix+'random_standardized_deltas']=random.numpy()
            edits[prefix+'null_standardized_deltas']=null.numpy()
    out.mkdir(parents=True)
    np.savez_compressed(out/'edits.npz',**edits)
    (out/'directions.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    metrics={key:paired_summary(rows,key) for key in ('targeted_shift','excess_over_random','excess_over_null',
             'target_decoder_ce_improvement','null_decoder_ce_improvement')}
    result=dict(config=vars(args),checkpoint_sha256=summary['checkpoint_sha256'],metrics=metrics,
                pairs=len(pairs),directions=len(rows),
                interpretation='Development diagnostic only. Decoder-only optimization; frozen policy never used in loss. Identical current observation/pose kept as decoder inputs. Radius in training-standardized hidden coordinates; controls have matching standardized L2 norm. Null control removes first-order centered-logit sensitivity, not all nonlinear belief effects. Paired bootstrap averages both directions; random draws are not independent experimental units. Restore check is deterministic reversibility, not behavioral rescue. Pose encoded inside hidden state and general navigation are NOT demonstrated preserved; off-manifold edits and decoder exploitation remain possible. Reserved episodes untouched.')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    plot(result,out)
    print(json.dumps(result,indent=2))


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,keys,labels in [
        (axes[0],['target_decoder_ce_improvement','null_decoder_ce_improvement'],['Targeted edit','Decoder-null control']),
        (axes[1],['targeted_shift','excess_over_random','excess_over_null'],['Targeted shift','Beyond random','Beyond decoder-null'])]:
        ax.axhline(0,color='gray',linestyle='--')
        for i,key in enumerate(keys):
            m=result['metrics'][key]
            if m['mean'] is not None:
                ax.plot(i,m['mean'],'o',color='C0')
                if m['ci95']: ax.plot([i,i],m['ci95'],color='C0')
        ax.set(xticks=range(len(keys)),xticklabels=labels)
        ax.tick_params(axis='x',labelrotation=15)
    axes[0].set(title='Did the decoder move toward the target?',ylabel='Target cross-entropy reduction (nats)')
    axes[1].set(title='Did the policy move as predicted?',ylabel='Change in target − source action probability')
    fig.suptitle(f"Development memory edits: {result['pairs']} pairs; paired 95% bootstrap intervals")
    fig.savefig(out/'interventions.png',dpi=180)
    fig.savefig(out/'interventions.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
