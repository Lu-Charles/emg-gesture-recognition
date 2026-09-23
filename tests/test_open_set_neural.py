import unittest
import numpy as np
import torch
from src.open_set_neural import remap_known,compactness,triplet,inconsistency,OpenSetNet


class NeuralRisks(unittest.TestCase):
    def test_unknown_label_guard(self):
        np.testing.assert_array_equal(remap_known([10,16]),[0,6])
        for y in [[9],[17],[],[10,0]]:
            with self.assertRaises(ValueError):remap_known(y)

    def test_compactness_literal_branches(self):
        z=torch.tensor([[.3,.4],[2.,0.]],dtype=torch.double)
        p=torch.zeros(7,2,dtype=torch.double);y=torch.tensor([0,1])
        self.assertAlmostEqual(compactness(z,p,y).item(),(.25+1.5)/2)

    def test_inconsistency_numpy_and_detached_positive(self):
        a=torch.tensor([[2.,1.,0.,-1.,-2.,-3.,-4.]],dtype=torch.double,requires_grad=True)
        b=torch.tensor([[2.,-4.,-3.,-2.,-1.,0.,1.]],dtype=torch.double,requires_grad=True)
        y=torch.tensor([0]);loss=inconsistency(a,b,y)
        def probs(v):
            s=np.maximum(v[0]-v[1:]-.5,0);e=np.exp(s-s.max());return e/e.sum()
        pa,pb=probs(a.detach().numpy()[0]),probs(b.detach().numpy()[0])
        self.assertAlmostEqual(loss.item(),-np.log(np.sum(pa*(1-pb)+pb*(1-pa))))
        loss.backward();self.assertEqual(a.grad[0,0].item(),0);self.assertEqual(b.grad[0,0].item(),0)
        self.assertGreater(a.grad[0,1:].abs().sum().item(),0)

    def test_triplet_nearest_wrong_class(self):
        p=torch.tensor([[0.,0.],[2.,0.],[4.,0.],[6.,0.],[8.,0.],[10.,0.],[12.,0.]])
        z=torch.tensor([[1.5,0.]]);y=torch.tensor([0])
        self.assertAlmostEqual(triplet(z,p,y).item(),2.)

    def test_seed_independence_and_update(self):
        torch.set_num_threads(2);m=OpenSetNet('predin');n=OpenSetNet('predin')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),n.parameters())))
        self.assertFalse(torch.equal(m.branches[0].prototypes,m.branches[1].prototypes))
        rng=np.random.default_rng(42);x=torch.tensor(rng.normal(size=(14,16,512)),dtype=torch.float32);y=torch.arange(14)%7
        before=[p.detach().clone() for p in m.parameters()];loss,_=m.loss(x,y);loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        with torch.no_grad():
            for p in m.parameters():p-=1e-4*p.grad
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,m.parameters())))
        pred,scores=m.predict_scores(x);self.assertTrue(torch.all((pred>=10)&(pred<=16)))
        expected=torch.stack([b(x)[1] for b in m.branches]).mean(0)
        torch.testing.assert_close(scores['prototype'],expected.max(1).values)


if __name__=='__main__':unittest.main()
