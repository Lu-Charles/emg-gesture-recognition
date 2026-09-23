import unittest
import numpy as np
import torch
from src.senic_neural import SeNicCNN,augment_channels,training_scale,fit,probabilities


class NeuralRisks(unittest.TestCase):
    def test_rotation_does_not_mix_windows_or_time(self):
        x=torch.arange(3*8*50).reshape(3,8,50)
        shifts=torch.tensor([-1,0,3]);z=augment_channels(x,shifts)
        for i,s in enumerate(shifts):self.assertTrue(torch.equal(z[i],torch.roll(x[i],int(s),dims=0)))
        self.assertTrue(torch.equal(augment_channels(z,-shifts),x))

    def test_scaler_only_uses_allowed_unique_samples(self):
        x=np.ones((3,400,8));x[2]=1000000
        self.assertEqual(training_scale(x,[0,1]),1.)
        self.assertEqual(training_scale(np.zeros_like(x),[0]),1e-8)

    def test_amplitude_and_all_labels_can_be_learned(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        x=np.zeros((28,8,50),dtype=np.float32);y=np.repeat(np.arange(7),4)
        for i,g in enumerate(y):x[i,g]=2
        model=SeNicCNN();result=fit(model,x,y,80,42)
        self.assertGreater(result['training_accuracy'],.99)
        p=probabilities(model,x);self.assertEqual(p.shape,(28,7))
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)


if __name__=='__main__':unittest.main()
