"""Regression checks for repeated substitution pairs and notebook descriptions."""
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from foldtree2.makesubmat import compute_pair_counts_and_bg, MatrixConvergenceMonitor
from production_alphabet_experiments import descriptions, notebook_functions
from foldtree2.src.encoder import RESIDUE_TRACK_ORDER, residue_features_for_encoder


class ProductionAlphabetTests(unittest.TestCase):
    def test_live_pdb_tracks_match_training_feature_order(self):
        graph = HeteroData()
        graph['res'].x = torch.zeros(2, 857)
        for i, name in enumerate(RESIDUE_TRACK_ORDER):
            graph[name].x = torch.full((2, 1), float(i))
        features = residue_features_for_encoder(graph, 865)
        torch.testing.assert_close(features[:, -8:], torch.arange(8).float().expand(2, -1))
        self.assertEqual(graph['res'].x.shape[1], 857)
        with self.assertRaises(ValueError):
            residue_features_for_encoder(graph, 866)

    def test_every_repeated_pair_and_first_record_is_counted(self):
        # One headerless record, with three occurrences of the same pair.
        row = 'q\tt\t0.2\t3\t0\t0\t1\t3\t1\t3\t1e-5\t10\tAAA\tAAA\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'allvall.csv'
            path.write_text(row)
            frame = pd.DataFrame({'seq': ['xxx', 'yyy']}, index=['q', 't'])
            counts, background = compute_pair_counts_and_bg([str(path)], frame, ['x', 'y'], {'x': 0, 'y': 1})
        np.testing.assert_array_equal(counts, [[0, 3], [0, 0]])
        np.testing.assert_array_equal(background, [3, 3])

    def test_description_preserves_unobserved_codebook_states(self):
        summary, rows = descriptions(['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'], ['AAAAA'] * 4,
                                     ['x', 'y'], notebook_functions())
        self.assertEqual(summary['alphabet_size'], 2)
        self.assertEqual(summary['observed_states'], 1)
        self.assertEqual(summary['effective_states'], 1)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(np.isfinite(row['backoff_entropy_rate_bits']) for row in rows))

    def test_convergence_requires_sustained_new_evidence(self):
        monitor = MatrixConvergenceMonitor(2, patience=2, min_files=2, min_pairs=10)
        matrix = np.ones((2, 2))
        monitor.update(1, matrix, pair_count=10)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])
        monitor.update(2, matrix, pair_count=20)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])
        monitor.update(3, matrix, pair_count=30)
        self.assertTrue(monitor.get_convergence_summary()['is_converged'])
        monitor.update(4, matrix, pair_count=30)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])

    def test_default_matches_single_update_notebook_threshold(self):
        monitor = MatrixConvergenceMonitor(2, min_files=2, min_pairs=10)
        matrix = np.ones((2, 2))
        monitor.update(1, matrix, pair_count=10)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])
        monitor.update(2, matrix + .002, pair_count=20)
        self.assertTrue(monitor.get_convergence_summary()['is_converged'])
        monitor.update(3, matrix + .02, pair_count=30)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])

    def test_updated_absolute_threshold(self):
        monitor = MatrixConvergenceMonitor(2, min_files=2, min_pairs=10)
        self.assertEqual(monitor.convergence_threshold, .025)
        matrix = np.ones((2, 2))
        monitor.update(1, matrix, pair_count=10)
        changed = matrix.copy()
        changed[0, 0] += .023167
        monitor.update(2, changed, pair_count=20)
        self.assertTrue(monitor.get_convergence_summary()['is_converged'])
        changed[0, 0] += .026
        monitor.update(3, changed, pair_count=30)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])

    def test_changing_matrix_and_missing_states_cannot_converge(self):
        monitor = MatrixConvergenceMonitor(2, patience=1, min_files=1, min_pairs=1)
        monitor.update(1, np.ones((2, 2)), pair_count=10)
        monitor.update(2, np.ones((2, 2)) * 2, pair_count=20)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])
        monitor.update(3, np.ones((2, 2)) * 2, pair_count=30, state_coverage=False)
        self.assertFalse(monitor.get_convergence_summary()['is_converged'])


if __name__ == '__main__':
    unittest.main()
