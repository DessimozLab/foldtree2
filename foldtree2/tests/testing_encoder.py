"""
Integration tests for encoder.py with mk2 pdb graphs.

These tests verify that an encoder model can be loaded and used to
encode PDB-derived graphs into FASTA token sequences.
"""

import os
import shutil
import tempfile
import unittest

import torch

from foldtree2.src.encoder import mk1_Encoder

try:
    from foldtree2.src.pdbgraphmkII import PDB2PyG
except ModuleNotFoundError:
    from foldtree2.src.pdbgraphmk2 import PDB2PyG


class TestEncoderIntegration(unittest.TestCase):
    """Test loading an encoder model and encoding mk2 graphs to FASTA."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.converter = PDB2PyG()

        self.pdb_a = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.761  -1.217  1.00  0.00           C
ATOM      6  N   GLY A   2       3.315   1.563   0.745  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.911   2.887   0.760  1.00  0.00           C
ATOM      8  C   GLY A   2       5.371   2.829   0.338  1.00  0.00           C
ATOM      9  O   GLY A   2       6.106   3.768   0.476  1.00  0.00           O
TER
END
"""

        self.pdb_b = """ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  SER A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   SER A   1       2.000   1.420   0.000  1.00  0.00           C
ATOM      4  O   SER A   1       1.230   2.390   0.000  1.00  0.00           O
ATOM      5  CB  SER A   1       1.980  -0.760  -1.210  1.00  0.00           C
ATOM      6  OG  SER A   1       1.400  -2.030  -1.280  1.00  0.00           O
ATOM      7  N   GLY A   2       3.300   1.560   0.740  1.00  0.00           N
ATOM      8  CA  GLY A   2       3.900   2.880   0.760  1.00  0.00           C
ATOM      9  C   GLY A   2       5.360   2.820   0.330  1.00  0.00           C
ATOM     10  O   GLY A   2       6.100   3.760   0.470  1.00  0.00           O
TER
END
"""

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_pdb(self, name, content):
        path = os.path.join(self.test_dir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    @staticmethod
    def _build_encoder(in_channels):
        metadata = {
            "edge_types": [
                ("res", "backbone", "res"),
                ("res", "backbonerev", "res"),
                ("res", "contactPoints", "res"),
                ("res", "window", "res"),
                ("res", "windowrev", "res"),
            ]
        }

        model = mk1_Encoder(
            in_channels=in_channels,
            hidden_channels=[32, 32],
            out_channels=16,
            num_embeddings=32,
            commitment_cost=0.25,
            metadata=metadata,
            flavor="transformer",
            edge_dim=12,
            EMA=True,
            concat_positions=False,
            learn_positions=False,
        )

        # Keep tests deterministic and CPU-safe even on GPU-enabled machines.
        model.device = torch.device("cpu")
        model.to(model.device)
        model.eval()
        return model

    @staticmethod
    def _load_torch_model(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def test_model_can_load_and_encode_pdbgraphs_to_fasta(self):
        """Save/load an encoder model and export encoded mk2 graphs as FASTA."""
        pdb_a_path = self._write_pdb("prot_a.pdb", self.pdb_a)
        pdb_b_path = self._write_pdb("prot_b.pdb", self.pdb_b)

        graph_a = self.converter.struct2pyg(pdb_a_path)
        graph_b = self.converter.struct2pyg(pdb_b_path)

        self.assertIsNotNone(graph_a)
        self.assertIsNotNone(graph_b)

        model = self._build_encoder(graph_a["res"].x.shape[1])

        ckpt_path = os.path.join(self.test_dir, "encoder.pt")
        torch.save(model, ckpt_path)

        loaded_model = self._load_torch_model(ckpt_path)
        loaded_model.device = torch.device("cpu")
        loaded_model.to(loaded_model.device)
        loaded_model.eval()

        with torch.no_grad():
            z, vq_loss = loaded_model(graph_a)
        self.assertEqual(z.shape[0], graph_a["res"].x.shape[0])
        self.assertEqual(z.shape[1], 16)
        self.assertIsNotNone(vq_loss)

        out_fasta = os.path.join(self.test_dir, "encoded.fasta")
        loaded_model.encode_structures_fasta([graph_a, graph_b], out_fasta, replace=True)

        self.assertTrue(os.path.exists(out_fasta))

        with open(out_fasta, "r", encoding="latin-1") as f:
            lines = [line.rstrip("\n") for line in f if line.rstrip("\n")]

        # Two sequences => 4 non-empty lines: >id, seq, >id, seq
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith(">"))
        self.assertTrue(lines[2].startswith(">"))

        self.assertEqual(lines[0][1:], graph_a.identifier)
        self.assertEqual(lines[2][1:], graph_b.identifier)

        self.assertEqual(len(lines[1]), graph_a["res"].x.shape[0])
        self.assertEqual(len(lines[3]), graph_b["res"].x.shape[0])


if __name__ == "__main__":
    unittest.main()
