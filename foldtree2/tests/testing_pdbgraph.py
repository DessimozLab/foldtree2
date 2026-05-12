"""
Tests for pdbgraphmk2.py - PDB to PyTorch Geometric graph conversion
This is CRITICAL for the entire pipeline
"""

import unittest
import os
import tempfile
import shutil
from importlib import import_module
import numpy as np
from foldtree2.src.config_paths import resolve_aapropcsv_path

try:
    PDB2PyG = import_module('foldtree2.src.pdbgraphmkII').PDB2PyG
except ModuleNotFoundError:
    PDB2PyG = import_module('foldtree2.src.pdbgraphmk2').PDB2PyG


class TestPDB2PyGInit(unittest.TestCase):
    """Test PDB2PyG initialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.aaprop_csv = resolve_aapropcsv_path(None)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        converter = PDB2PyG()
        self.assertIsNotNone(converter)
    
    def test_init_with_aapropcsv(self):
        """Test initialization with amino acid properties CSV"""
        if os.path.exists(self.aaprop_csv):
            converter = PDB2PyG(aapropcsv=self.aaprop_csv)
            self.assertIsNotNone(converter)
        else:
            self.skipTest("aaindex1.csv not found")


class TestPDBParsing(unittest.TestCase):
    """Test PDB file parsing functionality"""
    
    def setUp(self):
        """Set up test fixtures with sample PDB files"""
        self.test_dir = tempfile.mkdtemp()
        self.converter = PDB2PyG()
        
        # Create a simple test PDB file
        self.pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
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
        self.pdb_file = os.path.join(self.test_dir, 'test.pdb')
        with open(self.pdb_file, 'w') as f:
            f.write(self.pdb_content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_struct2pyg_basic(self):
        """Test basic structure to mk2 HeteroData conversion"""
        data = self.converter.struct2pyg(self.pdb_file)

        self.assertIsNotNone(data)

        # mk2 returns HeteroData with typed nodes/edges.
        self.assertIn('res', data.node_types)
        self.assertIn('AA', data.node_types)
        self.assertIn('coords', data.node_types)
        self.assertIn('chi', data.node_types)
        self.assertIn('sc_centroid', data.node_types)

        n_res = data['res'].x.shape[0]
        self.assertGreaterEqual(n_res, 2)
        self.assertEqual(tuple(data['AA'].x.shape), (n_res, 20))
        self.assertEqual(tuple(data['coords'].x.shape), (n_res, 3))
        self.assertEqual(tuple(data['chi'].x.shape), (n_res, 8))
        self.assertEqual(tuple(data['sc_centroid'].x.shape), (n_res, 4))
        self.assertEqual(tuple(data['positions'].x.shape), (n_res, 256))

        self.assertIn(('res', 'contactPoints', 'res'), data.edge_types)
        cp = data['res', 'contactPoints', 'res']
        self.assertEqual(cp.edge_index.shape[0], 2)
        self.assertEqual(cp.edge_attr.shape[1], 12)

        # Edge-attr format: [weight, relation_onehot_5, chem_bond_onehot_6].
        cp_attr = cp.edge_attr.detach().cpu().numpy()
        if cp_attr.shape[0] > 0:
            rel_onehot = cp_attr[:, 1:6]
            # After to_undirected coalescing, values can be accumulated (>1).
            self.assertTrue(np.all(np.count_nonzero(rel_onehot, axis=1) == 1))
            # relation index 2 => contactPoints => column offset 1 + 2 = 3
            self.assertTrue(np.all(rel_onehot[:, 2] > 0.0))
    
    def test_struct2pyg_missing_file(self):
        """Test handling of missing PDB file"""
        with self.assertRaises((FileNotFoundError, IOError)):
            self.converter.struct2pyg('/nonexistent/path/file.pdb')
    
    def test_struct2pyg_empty_file(self):
        """Test handling of empty PDB file"""
        empty_pdb = os.path.join(self.test_dir, 'empty.pdb')
        with open(empty_pdb, 'w') as f:
            f.write("")
        
        # Should handle gracefully (either return None or raise appropriate error)
        result = self.converter.struct2pyg(empty_pdb)
        self.assertIsNone(result)


class TestMultiChainPDB(unittest.TestCase):
    """Test handling of multi-chain PDB files"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.converter = PDB2PyG()
        
        # Create a multi-chain PDB file
        self.multichain_pdb = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  N   GLY B   1       5.000   0.000   0.000  1.00  0.00           N
ATOM      4  CA  GLY B   1       6.458   0.000   0.000  1.00  0.00           C
TER
END
"""
        self.pdb_file = os.path.join(self.test_dir, 'multichain.pdb')
        with open(self.pdb_file, 'w') as f:
            f.write(self.multichain_pdb)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_multichain_handling(self):
        """Test that multi-chain structures are handled without crashing"""
        data = self.converter.struct2pyg(self.pdb_file)

        self.assertIsNotNone(data)
        # mk2 picks the best polymer chain rather than concatenating chains.
        self.assertIn('res', data.node_types)
        self.assertGreaterEqual(data['res'].x.shape[0], 1)


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction from structures"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.converter = PDB2PyG()
        
        # Create test PDB
        self.pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
TER
END
"""
        self.pdb_file = os.path.join(self.test_dir, 'test.pdb')
        with open(self.pdb_file, 'w') as f:
            f.write(self.pdb_content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_node_features_exist(self):
        """Test that mk2 node features are extracted with expected shapes"""
        data = self.converter.struct2pyg(self.pdb_file)

        self.assertIsNotNone(data)
        self.assertIn('res', data.node_types)
        self.assertIn('AA', data.node_types)
        self.assertIn('bondangles', data.node_types)
        self.assertIn('plddt', data.node_types)
        self.assertIn('ss', data.node_types)

        n_res = data['res'].x.shape[0]
        self.assertEqual(data['AA'].x.shape[0], n_res)
        self.assertEqual(data['bondangles'].x.shape[1], 3)
        self.assertEqual(data['plddt'].x.shape[1], 1)
        self.assertEqual(data['ss'].x.shape[1], 3)
        self.assertEqual(data['res'].x.dim(), 2)

    def test_create_features_output_contract(self):
        """Test that create_features returns expected mk2 output keys and dimensions"""
        feat = self.converter.create_features(self.pdb_file)

        self.assertIsNotNone(feat)
        required_keys = [
            'aa', 'angles', 'bondangles', 'coords', 'cbcoords', 'ncoords',
            'ccoords', 'ocoords', 'contact_points', 'backbone', 'backbone_rev',
            'window', 'window_rev', 'ss', 'plddt', 'positional_encoding',
            'track_features', 'chi', 'sc_centroid'
        ]
        for key in required_keys:
            self.assertIn(key, feat)

        n_res = feat['coords'].shape[0]
        self.assertEqual(feat['aa'].shape, (n_res, 20))
        self.assertEqual(feat['bondangles'].shape, (n_res, 3))
        self.assertEqual(feat['positional_encoding'].shape, (n_res, 256))
        self.assertEqual(feat['chi'].shape, (n_res, 8))
        self.assertEqual(feat['sc_centroid'].shape, (n_res, 4))
    
    def test_edge_construction(self):
        """Test that mk2 relation edges and edge attributes are constructed"""
        data = self.converter.struct2pyg(self.pdb_file)

        self.assertIsNotNone(data)

        expected_relations = [
            ('res', 'backbone', 'res'),
            ('res', 'backbonerev', 'res'),
            ('res', 'contactPoints', 'res'),
            ('res', 'window', 'res'),
            ('res', 'windowrev', 'res'),
        ]
        for edge_type in expected_relations:
            self.assertIn(edge_type, data.edge_types)
            edge_store = data[edge_type]
            self.assertEqual(edge_store.edge_index.shape[0], 2)
            self.assertEqual(edge_store.edge_attr.shape[1], 12)

        # Godnode wiring should also exist.
        self.assertIn(('res', 'informs', 'godnode'), data.edge_types)
        self.assertIn(('godnode', 'informs', 'res'), data.edge_types)


class TestBackboneGeneration(unittest.TestCase):
    """Test backbone generation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = PDB2PyG()
    
    def test_get_backbone(self):
        """Test backbone generation for given chain length"""
        chainlen = 10
        
        backbone, backbone_rev = self.converter.get_backbone(chainlen)
        
        # Check shapes
        self.assertEqual(backbone.shape[0], chainlen)
        self.assertEqual(backbone.shape[1], chainlen)
        self.assertEqual(backbone_rev.shape[0], chainlen)
        self.assertEqual(backbone_rev.shape[1], chainlen)
        
        # Backbone should be sparse (mostly zeros)
        self.assertLess(np.count_nonzero(backbone), chainlen * chainlen)


class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = PDB2PyG()
    
    def test_get_positional_encoding(self):
        """Test positional encoding generation"""
        seq_len = 100
        encoding_dim = 256
        
        pos_encoding = self.converter.get_positional_encoding(seq_len, encoding_dim)
        
        # Check shape
        self.assertEqual(pos_encoding.shape, (seq_len, encoding_dim))
        
        # Values should be in reasonable range
        self.assertTrue(np.all(pos_encoding >= -1.0))
        self.assertTrue(np.all(pos_encoding <= 1.0))


class TestSparseMatrixConversion(unittest.TestCase):
    """Test sparse matrix conversion utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = PDB2PyG()
    
    def test_sparse2pairs(self):
        """Test sparse matrix to edge pairs conversion"""
        from scipy import sparse
        
        # Create a simple sparse adjacency matrix
        n = 5
        data = np.array([1, 1, 1, 1])
        row = np.array([0, 1, 2, 3])
        col = np.array([1, 2, 3, 4])
        sparse_mat = sparse.csr_matrix((data, (row, col)), shape=(n, n))
        
        pairs = self.converter.sparse2pairs(sparse_mat)
        
        # Should return edge pairs
        self.assertIsNotNone(pairs)
        self.assertEqual(pairs.shape, (2, 4))  # 2 x E for 4 edges


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.converter = PDB2PyG()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_very_short_chain(self):
        """Test handling of very short chains (1 residue)"""
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
TER
END
"""
        pdb_file = os.path.join(self.test_dir, 'short.pdb')
        with open(pdb_file, 'w') as f:
            f.write(pdb_content)
        
        # Should not crash
        data = self.converter.struct2pyg(pdb_file)
        self.assertIsNotNone(data)
    
    def test_missing_atoms(self):
        """Test handling of residues with missing atoms"""
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      3  N   GLY A   2       3.315   1.563   0.745  1.00  0.00           N
ATOM      4  CA  GLY A   2       3.911   2.887   0.760  1.00  0.00           C
TER
END
"""
        pdb_file = os.path.join(self.test_dir, 'missing_atoms.pdb')
        with open(pdb_file, 'w') as f:
            f.write(pdb_content)

        # Should not crash even with missing backbone atoms in one residue
        data = self.converter.struct2pyg(pdb_file)
        self.assertIsNotNone(data)


if __name__ == '__main__':
    unittest.main()