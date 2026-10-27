import unittest
from types import SimpleNamespace
import numpy as np
import torch
from analysis.compare_belief_decoders import fit_candidate, predict, region_targets, fresh_subset


class DecoderComparisonTest(unittest.TestCase):
    def test_test_features_labels_and_actions_do_not_affect_fit(self):
        rng=np.random.default_rng(0)
        x=rng.normal(size=(18,8))
        target=rng.dirichlet(np.ones(4),size=18)
        data=dict(split=np.repeat([0,1,2],6), belief=target.copy(), episode=np.repeat(np.arange(6),3))
        args=SimpleNamespace(parameter_budget=100,batch_size=3,epochs=3,patience=0,lr=.001)
        first=fit_candidate(x,target,data,args,0,'mlp',.01)
        x[12:]*=1000
        target[12:]=np.roll(target[12:],1,axis=1)
        data['belief'][12:]=target[12:]
        second=fit_candidate(x,target,data,args,0,'mlp',.01)
        self.assertEqual(first['history'],second['history'])
        for k in first['state_dict']:
            torch.testing.assert_close(first['state_dict'][k],second['state_dict'][k])
        kl,p=predict(first,x[:6],target[:6])
        self.assertTrue(np.isfinite(kl).all())
        np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)

    def test_quadrants_preserve_probability(self):
        cells=np.array([[1,1],[1,8],[8,1],[8,8]])
        b=np.array([[.1,.2,.3,.4]])
        np.testing.assert_allclose(region_targets(b,cells,10),b)

    def test_fresh_subset_excludes_old_and_duplicate_layouts(self):
        data=dict(episode=np.repeat(np.arange(4),2),
                  layout=np.repeat(np.array([[0],[1],[1],[2]]),2,axis=0),
                  split=np.zeros(8,int),cells=np.array([[1,1]]))
        result=fresh_subset(data,{(0,)},2)
        np.testing.assert_array_equal(np.unique(result['episode']),[1,3])
        self.assertTrue(np.all(result['split']==2))
        with self.assertRaises(ValueError):
            fresh_subset(data,{(0,)},3)


if __name__=='__main__':
    unittest.main()
