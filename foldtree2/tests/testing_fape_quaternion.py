import math
import unittest

import torch

from foldtree2.src.losses.fape import (
    quaternion_fape_loss,
    quaternion_geodesic_loss,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


class FapeQuaternionTests(unittest.TestCase):
    def test_quaternion_geodesic_is_zero_for_same_and_sign_flipped_quaternions(self):
        q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(torch.isclose(quaternion_geodesic_loss(q, q), torch.tensor(0.0)))
        self.assertTrue(torch.isclose(quaternion_geodesic_loss(-q, q), torch.tensor(0.0)))

    def test_quaternion_geodesic_returns_so3_angle_in_radians(self):
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        half_turn_x = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        loss = quaternion_geodesic_loss(half_turn_x, identity)

        self.assertTrue(torch.allclose(loss, torch.tensor(math.pi), atol=1e-6))

    def test_rotation_matrix_quaternion_round_trip_preserves_rotation(self):
        q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9238795, 0.3826834, 0.0, 0.0],
                [0.7071068, 0.0, 0.7071068, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )
        rot = quaternion_to_rotation_matrix(q)
        q_rt = rotation_matrix_to_quaternion(rot)
        rot_rt = quaternion_to_rotation_matrix(q_rt)

        self.assertTrue(torch.allclose(rot_rt, rot, atol=1e-5))

    def test_fape_is_zero_for_identical_frames(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 1.0, 0.0],
                [11.4, 1.0, 1.0],
            ]
        )

        loss = quaternion_fape_loss(q, origins, q, origins)

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-5))

    def test_fape_is_invariant_to_common_rigid_translation(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
        origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 1.0, 0.0],
            ]
        )
        offset = torch.tensor([100.0, -50.0, 25.0])

        loss = quaternion_fape_loss(q, origins, q, origins + offset)

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-4))

    def test_batched_fape_does_not_compare_across_structures(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [1000.0, 1000.0, 1000.0],
                [1003.8, 1000.0, 1000.0],
            ]
        )
        pred = origins.clone()
        pred[2:] += torch.tensor([50.0, 0.0, 0.0])
        batch = torch.tensor([0, 0, 1, 1])

        loss = quaternion_fape_loss(q, origins, q, pred, batch=batch)

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
