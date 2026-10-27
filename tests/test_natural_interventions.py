import unittest

import numpy as np
import torch
from torch import nn

from analysis.evaluate_natural_histories import select_pairs, stable_winner
from analysis.intervene_belief_memory import edit_memory, control_deltas, paired_summary


class NaturalInterventionTest(unittest.TestCase):
    def test_matching_is_exact_episode_disjoint_and_respects_allowed_rows(self):
        episodes=np.repeat(np.arange(8),3)
        observations=np.array([[e,7] if t==0 else [0,t] for e in range(8) for t in range(3)],dtype=np.float32)
        data=dict(episode=episodes,timestep=np.tile(np.arange(3),8),observation=observations,
                  pose=np.tile([.2,.3,1,0,0,0],(24,1)),
                  belief=np.array([[.5,.5,0,0] if e%2==0 else [0,0,.5,.5] for e in episodes]))
        allowed=episodes<6
        pairs=select_pairs(data,allowed,10,0)
        self.assertTrue(pairs)
        used=[]
        for a,b in pairs:
            self.assertTrue(allowed[a] and allowed[b])
            np.testing.assert_array_equal(data['observation'][a],data['observation'][b])
            self.assertEqual(data['timestep'][a],data['timestep'][b])
            used.extend([data['episode'][a],data['episode'][b]])
        self.assertEqual(len(set(used)),len(used))

    def test_stability_rejects_ties_changes_and_small_gaps(self):
        self.assertEqual(stable_winner([np.array([2.,1.,0]),np.array([3.,1.,0])],.1),0)
        self.assertIsNone(stable_winner([np.array([2.,2.,0])],.1))
        self.assertIsNone(stable_winner([np.array([2.,1.,0]),np.array([1.,2.,0])],.1))
        self.assertIsNone(stable_winner([np.array([2.,1.999,0])],.1))

    def test_edit_is_bounded_and_does_not_change_decoder_weights(self):
        decoder=nn.Linear(3,2,bias=False).requires_grad_(False)
        with torch.no_grad(): decoder.weight.copy_(torch.tensor([[1.,0.,0.],[-1.,0.,0.]]))
        before=decoder.weight.detach().clone()
        x=torch.zeros(3)
        target=torch.tensor([1.,0.])
        delta=edit_memory(decoder,x,target,2,.2,30)
        self.assertLessEqual(float(delta.norm()),.200001)
        self.assertGreater(float(delta[0]),0)
        torch.testing.assert_close(before,decoder.weight)

    def test_controls_match_norm_and_null_control_preserves_linear_logits(self):
        decoder=nn.Linear(4,2,bias=False).requires_grad_(False)
        with torch.no_grad(): decoder.weight.copy_(torch.tensor([[1.,0.,0.,0.],[-1.,0.,0.,0.]]))
        random,null,rank=control_deltas(decoder,torch.zeros(4),3,.2,10,torch.Generator().manual_seed(0))
        self.assertEqual(rank,1)
        torch.testing.assert_close(random.norm(dim=1),torch.full((10,),.2))
        torch.testing.assert_close(null.norm(dim=1),torch.full((10,),.2))
        torch.testing.assert_close(null[:,0],torch.zeros(10),atol=1e-6,rtol=0)

    def test_bidirectional_results_are_one_pair_not_two_samples(self):
        rows=[dict(pair_id=0,value=1.),dict(pair_id=0,value=-1.)]
        result=paired_summary(rows,'value')
        self.assertEqual(result['mean'],0.)
        self.assertEqual(result['pairs'],1)
        self.assertIsNone(result['ci95'])


if __name__=='__main__': unittest.main()
