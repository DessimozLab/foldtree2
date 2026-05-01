"""
Comprehensive tests for ft2treebuilder.py pipeline
Tests each step of the struct2tree pipeline with mocked models
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
import numpy as np
import torch

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from foldtree2.ft2treebuilder import treebuilder


class TestTreebuilderInit(unittest.TestCase):
    """Test treebuilder initialization"""
    
    def setUp(self):
        """Set up test fixtures with mocked model files"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, 'test_model')
        self.decoder_path = os.path.join(self.test_dir, 'test_decoder')
        
        # Create dummy model files
        with open(self.model_path + '.pt', 'wb') as f:
            f.write(b'dummy encoder')
        with open(self.decoder_path + '.pt', 'wb') as f:
            f.write(b'dummy decoder')
        
        self.mafftmat = os.path.join(self.test_dir, 'mafft_mat.mtx')
        self.submat = os.path.join(self.test_dir, 'submat.txt')
        with open(self.mafftmat, 'w') as f:
            f.write("dummy mafft matrix\n")
        with open(self.submat, 'w') as f:
            f.write("dummy submat\n")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    @patch('foldtree2.ft2treebuilder.torch.load')
    @patch('foldtree2.ft2treebuilder.PDB2PyG')
    def test_treebuilder_init_with_defaults(self, mock_pdb2pyg, mock_torch_load):
        """Test treebuilder initialization with default parameters"""
        # Mock the encoder and decoder objects
        mock_encoder = MagicMock()
        mock_encoder.num_embeddings = 256
        mock_encoder.device = 'cpu'
        
        mock_decoder = MagicMock()
        
        # Configure torch.load to return different objects based on path
        def load_side_effect(path, **kwargs):
            if 'encoder' in str(path) or path == self.model_path + '.pt':
                return mock_encoder
            else:
                return mock_decoder
        
        mock_torch_load.side_effect = load_side_effect
        mock_pdb2pyg.return_value = MagicMock()
        
        # Initialize treebuilder
        tb = treebuilder(
            model=self.model_path + '.pt',
            decoder_model=self.decoder_path + '.pt',
            mafftmat=self.mafftmat,
            submat=self.submat
        )
        
        # Verify initialization
        self.assertIsNotNone(tb.encoder)
        self.assertIsNotNone(tb.decoder)
        self.assertEqual(tb.raxml_path, None)
        self.assertFalse(tb.bs)
        self.assertFalse(tb.redo)
    
    @patch('foldtree2.ft2treebuilder.torch.load')
    @patch('foldtree2.ft2treebuilder.PDB2PyG')
    def test_treebuilder_init_with_charmaps(self, mock_pdb2pyg, mock_torch_load):
        """Test treebuilder initialization with character maps"""
        # Create a dummy charmaps pickle file
        import pickle
        charmaps_path = os.path.join(self.test_dir, 'charmaps.pkl')
        
        charmaps_data = {
            'pair_counts': np.array([[1, 2], [2, 1]]),
            'char_set': ['A', 'B', 'C'],
            'char_position_map': {'A': 0, 'B': 1, 'C': 2},
            'raxml_charset': '0 1 2'.split(),
            'raxml_char_position_map': {0: '0', 1: '1', 2: '2'},
            'background_freq': np.array([0.33, 0.33, 0.34]),
            'convergence_history': []
        }
        
        with open(charmaps_path, 'wb') as f:
            pickle.dump(charmaps_data, f)
        
        # Mock the encoder
        mock_encoder = MagicMock()
        mock_encoder.num_embeddings = 3
        mock_torch_load.return_value = mock_encoder
        mock_pdb2pyg.return_value = MagicMock()
        
        # Initialize with charmaps
        tb = treebuilder(
            model=self.model_path + '.pt',
            mafftmat=self.mafftmat,
            submat=self.submat,
            charmaps=charmaps_path
        )
        
        # Verify charmaps were loaded
        self.assertEqual(len(tb.alphabet), 3)
        self.assertEqual(tb.nchars, 3)


class TestStaticMethods(unittest.TestCase):
    """Test static methods in treebuilder class"""
    
    def test_formathex_3_char(self):
        """Test formathex with 3-character hex string"""
        result = treebuilder.formathex("0x1")
        self.assertEqual(result, "0x01")
    
    def test_formathex_4_char(self):
        """Test formathex with 4-character hex string"""
        result = treebuilder.formathex("0x1A")
        self.assertEqual(result, "0x1A")
    
    @patch('foldtree2.ft2treebuilder.subprocess.run')
    def test_run_mafft_textaln(self, mock_run):
        """Test MAFFT text alignment command"""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = treebuilder.run_mafft_textaln(
            infasta='/path/to/input.fasta',
            outaln='/path/to/output.aln',
            matrix='/path/to/matrix.mtx',
            mafft_path='mafft'
        )
        
        self.assertEqual(result, '/path/to/output.aln')
        mock_run.assert_called_once()
        
        # Verify the command contains expected components
        cmd = mock_run.call_args[0][0]
        self.assertIn('mafft', cmd)
        self.assertIn('--text', cmd)
        self.assertIn('/path/to/input.fasta', cmd)
    
    @patch('foldtree2.ft2treebuilder.subprocess.run')
    def test_mafft_hex2ascii(self, mock_run):
        """Test hex to ASCII conversion"""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = treebuilder.mafft_hex2ascii(
            intext='/path/to/input.hex',
            outfile='/path/to/output.ASCII',
            hex2text_path='./hex2maffttext'
        )
        
        self.assertEqual(result, '/path/to/output.ASCII')
        mock_run.assert_called_once()
    
    @patch('foldtree2.ft2treebuilder.subprocess.run')
    def test_fasta2hex(self, mock_run):
        """Test fasta to hex conversion"""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = treebuilder.fasta2hex(
            intext='/path/to/input.fasta',
            outfile='/path/to/output.hex',
            maffttext2hex='./maffttext2hex'
        )
        
        self.assertEqual(result, '/path/to/output.hex')
        mock_run.assert_called_once()
    
    @patch('foldtree2.ft2treebuilder.subprocess.run')
    def test_normal_mafft(self, mock_run):
        """Test normal MAFFT alignment"""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = treebuilder.normal_mafft(
            infasta='/path/to/input.fasta',
            outaln='/path/to/output.aln'
        )
        
        self.assertEqual(result, '/path/to/output.aln')
        mock_run.assert_called_once()
    
    def test_aln2dcict(self):
        """Test alignment file to dictionary conversion"""
        # Create a temporary alignment file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.aln', delete=False) as tmp:
            tmp.write(">SEQ1\n")
            tmp.write("AAAA\n")
            tmp.write(">SEQ2\n")
            tmp.write("CCCC\n")
            tmp_path = tmp.name
        
        try:
            result = treebuilder.aln2dcict(tmp_path)
            
            self.assertIn('SEQ1', result)
            self.assertIn('SEQ2', result)
            self.assertIn('AAAA', result['SEQ1'])
            self.assertIn('CCCC', result['SEQ2'])
        finally:
            os.unlink(tmp_path)


class TestStructSequenceConversion(unittest.TestCase):
    """Test structure to sequence conversion"""
    
    @patch('foldtree2.ft2treebuilder.PDB')
    def test_struct2sequence(self, mock_pdb):
        """Test structure to sequence extraction"""
        # Mock PDB parser
        mock_parser = MagicMock()
        mock_pdb.PDBParser.return_value = mock_parser
        
        # Mock structure object
        mock_structure = MagicMock()
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_residue = MagicMock()
        
        # Set up the mock hierarchy
        mock_structure.__iter__ = lambda self: iter([mock_model])
        mock_model.__iter__ = lambda self: iter([mock_chain])
        mock_chain.__iter__ = lambda self: iter([mock_residue])
        
        # Mock is_aa to return True
        mock_pdb.is_aa.return_value = True
        mock_residue.get_resname.return_value = 'ALA'
        
        mock_parser.get_structure.return_value = mock_structure
        
        # Create a temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write("SEQRES 1 A 1 ALA\n")
            tmp_path = tmp.name
        
        try:
            result = treebuilder.struct2sequence(tmp_path)
            
            self.assertEqual(result, 'ALA')
        finally:
            os.unlink(tmp_path)


class TestEncodingMethods(unittest.TestCase):
    """Test encoding related methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock encoder
        self.mock_encoder = MagicMock()
        self.mock_encoder.num_embeddings = 256
        self.mock_encoder.device = 'cpu'
        
        # Create mock converter
        self.mock_converter = MagicMock()
        
        # Create mock decoder
        self.mock_decoder = MagicMock()
        
        # Create treebuilder instance with mocked dependencies
        self.tb = MagicMock(spec=treebuilder)
        self.tb.encoder = self.mock_encoder
        self.tb.converter = self.mock_converter
        self.tb.decoder = self.mock_decoder
        self.tb.device = torch.device('cpu')
        self.tb.map = {chr(i): i-1 for i in range(1, 257)}
        self.tb.revmap = {i-1: chr(i) for i in range(1, 257)}
        self.tb.raxml_indices = {i: str(i) for i in range(256)}
        self.tb.rev_raxml_indices = {str(i): i for i in range(256)}
        self.tb.replace_dict = {chr(0): chr(246)}
        self.tb.replace_dict_ord = {ord(k): ord(v) for k, v in self.tb.replace_dict.items()}
        self.tb.formathex = treebuilder.formathex
        self.tb.nchars = 256
        self.tb.ordset = set(range(1, 257))
        
        # Create input directory with dummy PDB files
        self.input_dir = os.path.join(self.test_dir, 'input')
        os.makedirs(self.input_dir)
        for i in range(5):
            pdb_path = os.path.join(self.input_dir, f'prot{i}.pdb')
            with open(pdb_path, 'w') as f:
                f.write(f"SEQRES 1 A 10 {'A'*10}\n")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_replace_sp_chars(self):
        """Test special character replacement in encoded fasta"""
        # Create a test fasta file with special characters
        input_fasta = os.path.join(self.test_dir, 'input.fasta')
        output_fasta = os.path.join(self.test_dir, 'output.fasta')
        
        with open(input_fasta, 'w') as f:
            f.write(">SEQ1\n")
            f.write("A\x00BC\n")  # Contains null character
            f.write(">SEQ2\n")
            f.write("DEFG\n")
        
        # Bind the actual method to our mock
        treebuilder.replace_sp_chars(self.tb, input_fasta, output_fasta)
        
        # Verify output file exists
        self.assertTrue(os.path.exists(output_fasta))
        
        # Read and verify content
        with open(output_fasta, 'r') as f:
            content = f.read()
        
        self.assertIn('SEQ1', content)
        self.assertIn('SEQ2', content)
    
    def test_encodedfasta2hex(self):
        """Test encoded fasta to hex conversion"""
        input_fasta = os.path.join(self.test_dir, 'input.fasta')
        output_hex = os.path.join(self.test_dir, 'output.hex')
        
        with open(input_fasta, 'w') as f:
            f.write(">SEQ1\n")
            f.write("ABCD\n")
            f.write(">SEQ2\n")
            f.write("EFGH\n")
        
        # Bind the actual method to our mock
        treebuilder.encodedfasta2hex(self.tb, input_fasta, output_hex)
        
        # Verify output file exists
        self.assertTrue(os.path.exists(output_hex))
        
        # Read and verify hex format
        with open(output_hex, 'r') as f:
            lines = f.readlines()
        
        # First line should be header
        self.assertIn('>SEQ1', lines[0])
        # Second line should contain bare hex bytes (no 0x prefix)
        hex_payload = lines[1].strip().replace(' ', '')
        self.assertTrue(len(hex_payload) > 0)
        self.assertTrue(all(ch in '0123456789abcdefABCDEF' for ch in hex_payload))


class TestAlignmentReading(unittest.TestCase):
    """Test alignment reading and conversion methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock treebuilder with necessary attributes
        self.tb = MagicMock(spec=treebuilder)
        self.tb.map = {'A': 0, 'B': 1, 'C': 2}
        self.tb.revmap = {0: 'A', 1: 'B', 2: 'C'}
        self.tb.raxml_indices = {0: '0', 1: '1', 2: '2'}
        self.tb.rev_raxml_indices = {'0': 0, '1': 1, '2': 2}
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_read_textaln(self):
        """Test reading text alignment file"""
        # Create a test hex alignment file
        input_hex = os.path.join(self.test_dir, 'input.hex')
        output_fasta = os.path.join(self.test_dir, 'output.fasta')
        
        with open(input_hex, 'w') as f:
            f.write(">SEQ1\n")
            f.write("41 42 43 44\n")  # ABCD in hex
            f.write(">SEQ2\n")
            f.write("45 46 47 48\n")  # EFGH in hex
        
        # Bind the actual method to our mock
        result = treebuilder.read_textaln(self.tb, input_hex, output_fasta)
        
        self.assertEqual(result, output_fasta)
        self.assertTrue(os.path.exists(output_fasta))


class TestRAxMLMethods(unittest.TestCase):
    """Test RAxML-NG related methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock treebuilder
        self.tb = MagicMock(spec=treebuilder)
        self.tb.raxml_path = 'raxml-ng'
        self.tb.raxmlng_path = 'raxml-ng'
        self.tb.nchars = 256
        self.tb.ncores = 8
        self.tb.bs = False
        self.tb.redo = False
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    @patch('foldtree2.ft2treebuilder.subprocess.run')
    def test_run_raxml_ng(self, mock_run):
        """Test RAxML-NG execution"""
        mock_run.return_value = MagicMock(returncode=0)
        
        fasta_file = os.path.join(self.test_dir, 'input.fasta')
        matrix_file = os.path.join(self.test_dir, 'matrix.txt')
        output_prefix = os.path.join(self.test_dir, 'output')
        
        # Create dummy files
        with open(fasta_file, 'w') as f:
            f.write(">SEQ1\nAAAA\n")
        with open(matrix_file, 'w') as f:
            f.write("dummy matrix\n")
        
        result = treebuilder.run_raxml_ng(
            self.tb,
            fasta_file,
            matrix_file,
            256,
            output_prefix,
            iterations=10,
            cores=8
        )
        
        expected_result = output_prefix + '.raxml.bestTree'
        self.assertEqual(result, expected_result)
        mock_run.assert_called_once()
        
        # Verify command contains expected components
        cmd = mock_run.call_args[0][0]
        self.assertIn('--all', cmd)
        self.assertIn('--msa', cmd)
        self.assertIn(fasta_file, cmd)
    
    @patch('foldtree2.ft2treebuilder.subprocess.run')
    def test_run_raxml_ng_with_bootstrapping(self, mock_run):
        """Test RAxML-NG with bootstrapping enabled"""
        mock_run.return_value = MagicMock(returncode=0)

        fasta_file = os.path.join(self.test_dir, 'input_bs.fasta')
        matrix_file = os.path.join(self.test_dir, 'matrix_bs.txt')
        output_prefix = os.path.join(self.test_dir, 'output_bs')

        with open(fasta_file, 'w') as f:
            f.write(">SEQ1\nAAAA\n")
        with open(matrix_file, 'w') as f:
            f.write("dummy matrix\n")

        self.tb.bs = True

        result = treebuilder.run_raxml_ng(
            self.tb,
            fasta_file,
            matrix_file,
            256,
            output_prefix,
            iterations=25,
            cores=8
        )

        expected_result = output_prefix + '.raxml.bestTree'
        self.assertEqual(result, expected_result)
        mock_run.assert_called_once()

        cmd = mock_run.call_args[0][0]
        self.assertIn('--all', cmd)
        self.assertIn('--msa', cmd)
        self.assertIn('--bs-trees 25', cmd)
        self.assertIn('--bs-metric fbp', cmd)
        self.assertIn(fasta_file, cmd)