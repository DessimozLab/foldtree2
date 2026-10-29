import unittest

import torch
from torch_geometric.data import HeteroData

from foldtree2.src.losses.fape import (
    backbone_dihedrals_from_n_ca_c,
    ca_local_step_targets,
    ca_bond_length_loss,
    ca_pairwise_distance_loss,
    ca_step_loss,
    coarse_backbone_dihedrals_from_ca_frames,
    coarse_backbone_atoms_from_ca_frames,
    coarse_backbone_fape_loss,
    coarse_ca_loss,
    equivariant_ca_frame_rotmat,
    integrate_ca_steps,
    integrate_local_ca_steps,
    shift_prev_valid,
)
from foldtree2.src.mono_decoders import MultiMonoDecoder, Transformer_Geometry_Decoder
from foldtree2.learn_geometry_lightning import GeometryFocusedModule
from foldtree2.src.se3_struct_decoder import StagedTransformerRefiner, se3_denoiser


class TestCoarseCALoss(unittest.TestCase):
    def test_atom_graph_has_mandatory_covalent_edges(self):
        adj = se3_denoiser._covalent_atom_contacts(num_residues=3, atoms_per_residue=4, device=torch.device("cpu"))
        ca_idx, c_idx, cb_idx, n_idx = 0, 1, 2, 3

        for residue_idx in range(3):
            base = residue_idx * 4
            for src_atom, dst_atom in ((n_idx, ca_idx), (ca_idx, c_idx), (ca_idx, cb_idx)):
                src = base + src_atom
                dst = base + dst_atom
                self.assertTrue(adj[src, dst])
                self.assertTrue(adj[dst, src])

        for residue_idx in range(2):
            src = residue_idx * 4 + c_idx
            dst = (residue_idx + 1) * 4 + n_idx
            self.assertTrue(adj[src, dst])
            self.assertTrue(adj[dst, src])

    def test_staged_step_coordinate_round_trip(self):
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        batch = torch.tensor([0, 0, 0, 1, 1])
        steps = StagedTransformerRefiner._coords_to_steps(coords, batch)
        rebuilt = StagedTransformerRefiner._coords_from_steps(steps, batch)
        self.assertTrue(torch.allclose(rebuilt, coords))

    def test_derived_frames_rotate_with_coordinates(self):
        generator = torch.Generator().manual_seed(17)
        random_matrix = torch.randn(3, 3, generator=generator)
        q, _ = torch.linalg.qr(random_matrix)
        if torch.linalg.det(q) < 0:
            q[:, 0] *= -1

        ca = torch.stack([
            torch.arange(7, dtype=torch.float32),
            torch.sin(torch.arange(7, dtype=torch.float32)),
            torch.cos(torch.arange(7, dtype=torch.float32)),
        ], dim=-1)
        n = ca + torch.tensor([[-0.5, 0.8, 0.2]]).expand_as(ca)
        c = ca + torch.tensor([[1.4, -0.3, 0.6]]).expand_as(ca)

        ca_rot = ca @ q.T
        n_rot = n @ q.T
        c_rot = c @ q.T

        R, t, _, _ = GeometryFocusedModule._frames_from_ca_only(ca)
        R_rot, t_rot, _, _ = GeometryFocusedModule._frames_from_ca_only(ca_rot)
        self.assertTrue(torch.allclose(R_rot, q @ R, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(t_rot, t @ q.T, atol=1e-5, rtol=1e-5))

        nca_R = GeometryFocusedModule._frames_from_n_ca_c(n, ca, c)
        nca_R_rot = GeometryFocusedModule._frames_from_n_ca_c(n_rot, ca_rot, c_rot)
        self.assertTrue(torch.allclose(nca_R_rot, q @ nca_R, atol=1e-5, rtol=1e-5))

    def test_two_residue_chain_flags_twist_undefined(self):
        # Every residue of a two-residue chain has perfectly collinear
        # neighbor-extrapolated segments, so CA geometry alone cannot fix a
        # twist about the tangent for either residue.
        ca = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])
        rot, twist_undefined = equivariant_ca_frame_rotmat(ca)
        self.assertTrue(torch.equal(twist_undefined, torch.tensor([True, True])))
        # The frame must still be a valid (numerically stable) orthonormal basis.
        identity = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
        self.assertTrue(torch.allclose(rot @ rot.transpose(-1, -2), identity, atol=1e-5))

    def test_straight_chain_flags_only_collinear_residues(self):
        # A perfectly straight three-residue chain is collinear everywhere, but
        # a kinked one should only flag the undefined residue(s).
        straight = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        _, straight_undefined = equivariant_ca_frame_rotmat(straight)
        self.assertTrue(straight_undefined.all())

        kinked = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        _, kinked_undefined = equivariant_ca_frame_rotmat(kinked)
        self.assertFalse(kinked_undefined[1])

    def test_ca_step_loss_requires_both_endpoints_confident(self):
        # Residues 0..3; only residue 2 is low-confidence.
        true_ca = torch.tensor([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ])
        pred_steps = torch.zeros_like(true_ca)
        pred_steps[1:] = true_ca[1:] - true_ca[:-1]
        plddt = torch.tensor([1.0, 1.0, 0.0, 1.0])

        # Steps landing at 1 (0->1) and arriving via 2 (1->2, 2->3) all touch
        # residue 2 as an endpoint except step 0->1; only that step should count.
        loss = ca_step_loss(pred_steps, true_ca, plddt=plddt, plddt_thresh=0.5)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-5))

        # Perturb only the step that should be excluded (landing at residue 2);
        # the masked loss must stay zero even though the raw steps disagree there.
        bad_steps = pred_steps.clone()
        bad_steps[2] += 5.0
        masked_loss = ca_step_loss(bad_steps, true_ca, plddt=plddt, plddt_thresh=0.5)
        self.assertTrue(torch.isclose(masked_loss, torch.tensor(0.0), atol=1e-5))
        unmasked_loss = ca_step_loss(bad_steps, true_ca)
        self.assertGreater(float(unmasked_loss), 0.1)

    def test_ca_bond_and_pairwise_losses_require_both_endpoints_confident(self):
        true_ca = torch.tensor([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ])
        pred_steps = torch.zeros_like(true_ca)
        pred_steps[1:] = true_ca[1:] - true_ca[:-1]
        pred_steps[2] = torch.tensor([100.0, 0.0, 0.0])  # only touches residue 2
        plddt = torch.tensor([1.0, 1.0, 0.0, 1.0])

        bond_masked = ca_bond_length_loss(pred_steps, plddt=plddt, plddt_thresh=0.5)
        self.assertTrue(torch.isclose(bond_masked, torch.tensor(0.0), atol=1e-5))
        bond_unmasked = ca_bond_length_loss(pred_steps)
        self.assertGreater(float(bond_unmasked), 1.0)

        pred_ca = true_ca.clone()
        pred_ca[2] += 100.0
        pairwise_masked = ca_pairwise_distance_loss(pred_ca, true_ca, min_seq_sep=1, plddt=plddt, plddt_thresh=0.5)
        self.assertTrue(torch.isclose(pairwise_masked, torch.tensor(0.0), atol=1e-5))
        pairwise_unmasked = ca_pairwise_distance_loss(pred_ca, true_ca, min_seq_sep=1)
        self.assertGreater(float(pairwise_unmasked), 1.0)

    def test_shift_prev_valid_marks_chain_starts_false(self):
        valid = torch.tensor([True, True, False, True])
        batch = torch.tensor([0, 0, 1, 1])
        prev_valid = shift_prev_valid(valid, batch_idx=batch)
        # Chain starts (residues 0 and 2) have no previous residue.
        self.assertEqual(prev_valid.tolist(), [False, True, False, False])

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

    def test_coarse_backbone_atoms_follow_ca_and_frames(self):
        ca = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frames = torch.stack([torch.eye(3), self._rot_z(0.5)])

        atoms = coarse_backbone_atoms_from_ca_frames(ca, frames)

        self.assertEqual(tuple(atoms.shape), (2, 3, 3))
        self.assertTrue(torch.allclose(atoms[:, 0], ca))
        self.assertTrue(torch.isfinite(atoms).all())
        self.assertGreater(float((atoms[:, 1] - ca).norm(dim=-1).mean()), 1.0)
        self.assertGreater(float((atoms[:, 2] - ca).norm(dim=-1).mean()), 1.0)

    def test_coarse_backbone_fape_is_zero_for_identical_atoms(self):
        ca = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.5, 0.0],
            ]
        )
        frames = torch.stack([torch.eye(3), self._rot_z(0.2), self._rot_z(-0.1)])
        atoms = coarse_backbone_atoms_from_ca_frames(ca, frames)

        loss = coarse_backbone_fape_loss(atoms, atoms, frames, frames, ca, ca)

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_backbone_dihedrals_from_n_ca_c_masks_chain_ends(self):
        n = torch.tensor(
            [
                [-0.5, 1.0, 0.0],
                [3.3, 1.1, 0.2],
                [7.1, 1.0, -0.2],
            ]
        )
        ca = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.2, 0.0],
            ]
        )
        c = torch.tensor(
            [
                [1.5, 0.0, 0.1],
                [5.3, 0.2, -0.1],
                [9.1, 0.3, 0.2],
            ]
        )

        angles, mask = backbone_dihedrals_from_n_ca_c(n, ca, c)

        self.assertEqual(tuple(angles.shape), (3, 3))
        self.assertEqual(tuple(mask.shape), (3, 3))
        self.assertTrue(torch.isfinite(angles).all())
        self.assertTrue(torch.equal(mask[:, 0], torch.tensor([False, True, True])))
        self.assertTrue(torch.equal(mask[:, 1], torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(mask[:, 2], torch.tensor([True, True, False])))

    def test_coarse_backbone_dihedrals_from_frames_are_finite(self):
        ca = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.5, 0.0],
                [11.2, 0.8, 0.2],
            ]
        )
        frames = torch.stack([torch.eye(3), self._rot_z(0.2), self._rot_z(-0.1), self._rot_z(0.4)])

        angles, mask = coarse_backbone_dihedrals_from_ca_frames(ca, frames)

        self.assertEqual(tuple(angles.shape), (4, 3))
        self.assertTrue(torch.isfinite(angles).all())
        self.assertEqual(int(mask.sum()), 9)

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
