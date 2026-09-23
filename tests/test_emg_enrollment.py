import unittest
import numpy as np
import torch
from src.emg_cnn import CompactEMGNet
from src.emg_enrollment import enrollment_path
from scripts.pilot_emg_cnn import fit


class EnrollmentPathTests(unittest.TestCase):
    def test_checkpoints_preserve_uninterrupted_adam_and_isolate_mutation(self):
        torch.set_num_threads(2)
        x = np.random.default_rng(2).normal(size=(9, 16, 512)).astype(np.float32)
        y = np.arange(9) % 3
        torch.manual_seed(42)
        model = CompactEMGNet()
        path = enrollment_path(model, x, y, [2, 4], .001, 42, "cpu")
        first = next(path)
        torch.manual_seed(42)
        reference = CompactEMGNet()
        fit(reference, x, y, 2, .001, 42, "cpu")
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(first["model"][name], value, rtol=0, atol=0)
        for value in first["model"].values():
            value.zero_()
        final = next(path)
        torch.manual_seed(42)
        reference = CompactEMGNet()
        history = fit(reference, x, y, 4, .001, 42, "cpu")
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(final["model"][name], value, rtol=0, atol=0)
        self.assertEqual(final["epoch_losses"], history["epoch_losses"])
        self.assertEqual(int(next(iter(final["optimizer"]["state"].values()))["step"]), 4)

    def test_ambiguous_checkpoint_sequence_is_rejected(self):
        for checkpoints in ([], [0], [2, 2], [4, 2]):
            with self.assertRaises(ValueError):
                next(enrollment_path(None, None, None, checkpoints, .001, 42, "cpu"))


if __name__ == "__main__":
    unittest.main()
