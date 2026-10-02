"""Public conversion routes must select mk2, including multiprocessing."""
import sys

import pytest

from foldtree2 import encode_pdbs
from foldtree2.scripts import pdbs_to_graphs
from foldtree2.src import pdbgraphmk2


def test_public_converter_is_mk2():
    assert pdbs_to_graphs.PDB2PyG is pdbgraphmk2.PDB2PyG
    assert encode_pdbs.pdbgraphmk2.PDB2PyG is pdbgraphmk2.PDB2PyG


@pytest.mark.parametrize('multiprocessing', [False, True])
@pytest.mark.parametrize('route', ['public', 'encode'])
def test_conversion_dispatches_to_mk2_methods(tmp_path, monkeypatch, route, multiprocessing):
    source = tmp_path / 'pdbs'
    source.mkdir()
    pdb = source / 'sample.pdb'
    pdb.write_text('fixture')
    output = tmp_path / 'graphs.h5'
    calls = []

    class Converter:
        def __init__(self, aapropcsv=None):
            pass

        def store_pyg(self, files, filename, **kwargs):
            calls.append(('serial', files, str(filename), kwargs))

        def store_pyg_mp_pool(self, files, filename, **kwargs):
            calls.append(('pool', files, str(filename), kwargs))

    if route == 'public':
        monkeypatch.setattr(pdbs_to_graphs, 'PDB2PyG', Converter)
        monkeypatch.setattr(sys, 'argv', ['pdbs-to-graphs', str(source), str(output)] +
                            (['--mp', '--ncpu', '2'] if multiprocessing else []))
        pdbs_to_graphs.main()
    else:
        monkeypatch.setattr(encode_pdbs.pdbgraphmk2, 'PDB2PyG', Converter)
        encode_pdbs.main([str(source), str(output)] +
                         (['--multiprocessing', '--ncpu', '2'] if multiprocessing else []))
    assert len(calls) == 1
    method, files, filename, options = calls[0]
    assert method == ('pool' if multiprocessing else 'serial')
    assert files == [str(pdb)]
    assert filename == str(output)
    if multiprocessing:
        assert options['ncpu'] == 2


def test_legacy_feature_request_is_not_silently_ignored(tmp_path):
    with pytest.raises(SystemExit):
        encode_pdbs.main([str(tmp_path), str(tmp_path / 'graphs.h5'), '--add-prody'])
