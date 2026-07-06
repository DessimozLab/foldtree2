import unittest

import torch
from torch_geometric.data import HeteroData

from foldtree2.src.losses.fape import coarse_ca_loss, integrate_ca_steps
from foldtree2.src.mono_decoders import MultiMonoDecoder


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

    def test_multi_decoder_coarse_ca_outputs_are_finite(self):
        data = HeteroData()
        data["res"].x = torch.randn(7, 12)
        data["positions"].x = torch.randn(7, 256)
        data["coords"].x = torch.stack(
            [
                torch.arange(7, dtype=torch.float32) * 3.8,
                torch.zeros(7),
                torch.zeros(7),
            ],
            dim=-1,
        )

        model = MultiMonoDecoder(
            {
                "coarse_ca": {
                    "in_channels": {"res": 12},
                    "hidden_dim": 16,
                    "layers": 1,
                    "nheads": 4,
                    "dropout": 0.0,
                }
            }
        )

        out = model(data)
        self.assertEqual(tuple(out["ca_steps_pred"].shape), (7, 3))
        self.assertEqual(tuple(out["ca_coords_pred"].shape), (7, 3))
        self.assertTrue(torch.isfinite(out["ca_steps_pred"]).all())
        self.assertTrue(torch.isfinite(out["ca_coords_pred"]).all())

        loss = coarse_ca_loss(
            out["ca_steps_pred"],
            data["coords"].x,
            pred_ca=out["ca_coords_pred"],
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
