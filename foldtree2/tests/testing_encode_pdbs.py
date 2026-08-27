import os
import sys
import tempfile
import types
import unittest

pdbgraph_stub = types.ModuleType('foldtree2.src.pdbgraph')
pdbgraph_stub.PDB2PyG = object
sys.modules.setdefault('foldtree2.src.pdbgraph', pdbgraph_stub)

pdbgraphmk2_stub = types.ModuleType('foldtree2.src.pdbgraphmk2')
pdbgraphmk2_stub.PDB2PyG = object
sys.modules.setdefault('foldtree2.src.pdbgraphmk2', pdbgraphmk2_stub)

from foldtree2.encode_pdbs import (
    _checkpoint_manifest_path,
    _filter_remaining_inputs,
    _load_checkpoint_state,
    _save_checkpoint_state,
)


class TestCheckpointHelpers(unittest.TestCase):
    def test_checkpoint_roundtrip_and_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_h5 = os.path.join(tmpdir, 'structures.h5')
            manifest_path = _checkpoint_manifest_path(output_h5)
            files = [
                os.path.join(tmpdir, 'a.pdb'),
                os.path.join(tmpdir, 'b.pdb'),
            ]

            self.assertEqual(_load_checkpoint_state(manifest_path), set())

            _save_checkpoint_state(manifest_path, {os.path.abspath(files[0])})
            self.assertEqual(_load_checkpoint_state(manifest_path), {os.path.abspath(files[0])})

            remaining = _filter_remaining_inputs(files, manifest_path)
            self.assertEqual(remaining, [os.path.abspath(files[1])])


if __name__ == '__main__':
    unittest.main()
