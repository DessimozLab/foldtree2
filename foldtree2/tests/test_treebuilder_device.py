"""Explicit device selection must override checkpoint runtime attributes."""
import unittest
from unittest.mock import patch

import torch

from foldtree2.ft2treebuilder import treebuilder


class TreebuilderDeviceTests(unittest.TestCase):
    def test_ancestral_uses_shared_runner_and_fitted_model(self):
        builder = treebuilder.__new__(treebuilder)
        builder.raxmlng_path = 'raxml-ng'
        builder.overwrite = True
        builder.log = lambda *args, **kwargs: None
        with patch('foldtree2.ft2treebuilder.run_ancestral',
                   return_value={'states': 'chosen_prefix.raxml.ancestralStates'}) as run:
            result = builder.run_raxml_ng_ancestral_struct(
                'arbitrary_alignment.fa', 'own_tree', 'matrix', 30, 'chosen_prefix',
                fitted_model='own_fitted_model')
        self.assertEqual(result, 'chosen_prefix.raxml.ancestralStates')
        self.assertEqual(run.call_args.args[2], 'own_fitted_model')
        self.assertTrue(run.call_args.kwargs['freeze_fitted'])

    def test_explicit_cpu_overrides_serialized_cuda_for_both_models(self):
        encoder = torch.nn.Linear(2, 2)
        encoder.num_embeddings = 4
        encoder.device = torch.device('cuda:0')
        decoder = torch.nn.Linear(2, 2)
        decoder.device = torch.device('cuda:0')
        with patch('foldtree2.ft2treebuilder.torch.load', side_effect=[encoder, decoder]), \
             patch('foldtree2.ft2treebuilder.PDB2PyG'):
            builder = treebuilder('encoder.pt', decoder_model='decoder.pt', device='cpu')
        self.assertEqual(builder.encoder.device, torch.device('cpu'))
        self.assertEqual(builder.decoder.device, torch.device('cpu'))
        self.assertEqual(next(builder.encoder.parameters()).device, torch.device('cpu'))
        self.assertEqual(next(builder.decoder.parameters()).device, torch.device('cpu'))
        self.assertFalse(builder.encoder.training)
        self.assertFalse(builder.decoder.training)


if __name__ == '__main__':
    unittest.main()
