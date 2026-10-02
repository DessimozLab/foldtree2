"""Regression tests for the approved notebook held-out protocol."""

import unittest

import torch

from foldtree2.src.training_protocol import held_out_split


class TrainingProtocolTests(unittest.TestCase):
    def test_matches_notebook_without_changing_global_rng(self):
        dataset = range(20089)
        before = torch.random.get_rng_state().clone()
        training, validation = held_out_split(dataset)
        expected = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7)).tolist()
        self.assertEqual(len(training), 18080)
        self.assertEqual(len(validation), 2009)
        self.assertEqual(validation.indices, expected[:2009])
        self.assertEqual(training.indices, expected[2009:])
        self.assertFalse(set(training.indices) & set(validation.indices))
        self.assertEqual(set(training.indices) | set(validation.indices), set(dataset))
        torch.testing.assert_close(before, torch.random.get_rng_state())

    def test_small_split_and_invalid_inputs(self):
        self.assertEqual(len(held_out_split(range(15))[1]), 2)
        self.assertEqual(len(held_out_split(range(2))[1]), 1)
        for dataset, fraction in [(range(1), .1), (range(10), 0), (range(10), 1)]:
            with self.assertRaises(ValueError):
                held_out_split(dataset, fraction)


if __name__ == '__main__':
    unittest.main()
