import unittest

import torch

from foldtree2.src.losses.fape import coarse_ca_loss, integrate_ca_steps


class TestCoarseCALoss(unittest.TestCase):
    def test_integrate_ca_steps_single_chain(self):
        steps = torch.zeros(5, 3)
        steps[1:, 0] = 3.8

        coords = integrate_ca_steps(steps)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.0, 0.0],
                [11.4, 0.0, 0.0],
                [15.2, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(coords, expected, atol=1e-5))

    def test_integrate_ca_steps_resets_per_batch(self):
        steps = torch.zeros(5, 3)
        steps[[1, 2, 4], 0] = 3.8
        batch = torch.tensor([0, 0, 0, 1, 1])

        coords = integrate_ca_steps(steps, batch_idx=batch)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(coords, expected, atol=1e-5))

    def test_coarse_ca_loss_is_zero_for_matching_trace(self):
        steps = torch.zeros(5, 3)
        steps[1:, 0] = 3.8
        coords = integrate_ca_steps(steps)

        loss, components = coarse_ca_loss(
            steps,
            coords,
            pred_ca=coords,
            return_components=True,
        )

        self.assertLess(float(loss), 1e-6)
        self.assertLess(float(components["step"]), 1e-6)
        self.assertLess(float(components["bond"]), 1e-6)
        self.assertLess(float(components["pairwise"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
