import unittest

import torch
from torch_geometric.data import HeteroData

from foldtree2.src.losses.fape import (
    ca_local_step_targets,
    coarse_ca_loss,
    integrate_ca_steps,
    integrate_local_ca_steps,
)
from foldtree2.src.mono_decoders import MultiMonoDecoder, Transformer_Geometry_Decoder


class TestCoarseCALoss(unittest.TestCase):
    def _rot_z(self, angle):
        c = torch.cos(torch.tensor(angle))
        s = torch.sin(torch.tensor(angle))
        return torch.tensor(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

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

    def test_local_ca_steps_round_trip_with_previous_frames(self):
        frames = torch.stack(
            [
                torch.eye(3),
                self._rot_z(0.3),
                self._rot_z(-0.2),
                self._rot_z(0.7),
            ]
        )
        local_steps = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.2, 0.0],
                [3.7, -0.4, 0.1],
                [3.6, 0.3, -0.2],
            ]
        )

        coords = integrate_local_ca_steps(local_steps, frames, frame_offset="prev")
        target_steps, mask = ca_local_step_targets(coords, frames, frame_offset="prev")

        self.assertTrue(torch.equal(mask, torch.tensor([False, True, True, True])))
        self.assertTrue(torch.allclose(target_steps, local_steps, atol=1e-5))

    def test_local_ca_steps_round_trip_resets_per_batch(self):
        frames = torch.stack(
            [
                torch.eye(3),
                self._rot_z(0.5),
                self._rot_z(-0.5),
                torch.eye(3),
                self._rot_z(1.0),
            ]
        )
        local_steps = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.7, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [3.6, -0.2, 0.1],
            ]
        )
        batch = torch.tensor([0, 0, 0, 1, 1])

        coords = integrate_local_ca_steps(local_steps, frames, batch_idx=batch, frame_offset="prev")
        target_steps, mask = ca_local_step_targets(coords, frames, batch_idx=batch, frame_offset="prev")

        self.assertTrue(torch.equal(mask, torch.tensor([False, True, True, False, True])))
        self.assertTrue(torch.allclose(target_steps, local_steps, atol=1e-5))

    def test_coarse_ca_loss_is_zero_for_matching_local_trace(self):
        frames = torch.stack([torch.eye(3), self._rot_z(0.4), self._rot_z(-0.2), self._rot_z(0.9)])
        local_steps = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.8, 0.1, 0.0],
                [3.7, -0.2, 0.2],
            ]
        )
        coords = integrate_local_ca_steps(local_steps, frames, frame_offset="prev")

        loss, components = coarse_ca_loss(
            local_steps,
            coords,
            frames=frames,
            frame_offset="prev",
            bond_weight=0.0,
            return_components=True,
        )

        self.assertLess(float(loss), 1e-5)
        self.assertLess(float(components["step"]), 1e-6)
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

    def test_transformer_geometry_decoder_has_separate_geometry_heads(self):
        data = HeteroData()
        data["res"].x = torch.randn(6, 8)
        data["positions"].x = torch.randn(6, 256)

        model = Transformer_Geometry_Decoder(
            in_channels={"res": 8},
            hidden_channels={("res", "backbone", "res"): [16, 16, 16]},
            concat_positions=True,
            nheads=4,
            layers=1,
            RTdecoder_hidden=[16, 12, 8],
            rotationdecoder_hidden=[16, 12, 8],
            castepdecoder_hidden=[16, 12, 8],
            ssdecoder_hidden=[16, 12, 8],
            anglesdecoder_hidden=[16, 12, 8],
            dropout=0.0,
            residual=False,
            output_rt=True,
            output_ca_steps=True,
            output_ss=True,
            output_angles=True,
        )

        out = model(data)

        self.assertIn("quat_head", model.head)
        self.assertIn("trans_head", model.head)
        self.assertIn("ca_step_head", model.head)
        self.assertIsNot(model.head["quat_head"], model.head["trans_head"])
        self.assertEqual(tuple(out["rt_pred"].shape), (6, 7))
        self.assertEqual(tuple(out["ca_step_pred"].shape), (6, 3))
        self.assertTrue(torch.isfinite(out["rt_pred"]).all())
        self.assertTrue(torch.isfinite(out["quat_pred"]).all())
        self.assertTrue(torch.isfinite(out["trans_pred"]).all())
        self.assertTrue(torch.isfinite(out["ca_step_pred"]).all())
        self.assertIsNot(out["quat_pred"], out["rt_pred"][..., :4])
        self.assertIsNot(out["trans_pred"], out["rt_pred"][..., 4:])
        self.assertIsNot(out["ca_step_pred"], out["rt_pred"][..., 4:])


if __name__ == "__main__":
    unittest.main()
