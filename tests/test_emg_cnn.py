import unittest
import numpy as np
import torch
from src.emg_cnn import ChannelStandardizer, CompactEMGNet


class NeuralRiskTests(unittest.TestCase):
    def test_scoring_amplitude_does_not_refit_normalization(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(12, 16, 512)).astype(np.float32)
        scale = ChannelStandardizer().fit(train)
        before = (scale.mean.copy(), scale.scale.copy())
        output = scale.transform(train * 100)
        np.testing.assert_array_equal(scale.mean, before[0])
        np.testing.assert_array_equal(scale.scale, before[1])
        self.assertGreater(float(np.std(output)), 90)

    def test_head_update_leaves_encoder_unchanged(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = CompactEMGNet()
        model.set_update_mode("head")
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 16, 512)), torch.arange(8))
        loss.backward()
        optimizer.step()
        for name, value in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, before["encoder." + name]))
        self.assertFalse(torch.equal(model.head.weight, before["head.weight"]))

    def test_full_update_reaches_encoder_and_frozen_updates_nothing(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = CompactEMGNet()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 16, 512)), torch.arange(8))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in model.encoder.parameters()))
        model.set_update_mode("frozen")
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
