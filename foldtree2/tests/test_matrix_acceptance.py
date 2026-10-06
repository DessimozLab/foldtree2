import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from prepare_production_alphabets import check_convergence_acceptance


def test_nonconverged_acceptance_is_explicit_and_bound_to_artifacts():
    payload = {'counting_method': 'headerless_all_pair_occurrences_v2',
               'convergence': {'is_converged': False}}
    hashes = {'encoder': 'enc', 'decoder': 'dec', 'mafft': 'msa', 'raxml': 'tree'}
    acceptance = {'status': 'accepted_nonconverged', 'authorization': 'explicit_user_approval',
                  'reason': 'User requested continuing benchmarks', 'approved_at_utc': '2026-10-06',
                  'artifact_sha256': hashes, 'metrics_sha256': 'metrics'}
    assert check_convergence_acceptance(payload, acceptance, hashes, 'metrics') == 'accepted_nonconverged'
    assert payload['convergence']['is_converged'] is False
    for bad in [None, {}, {**acceptance, 'authorization': 'automatic'},
                {**acceptance, 'metrics_sha256': 'changed'}, {**acceptance, 'reason': ''},
                {**acceptance, 'artifact_sha256': {**hashes, 'raxml': 'changed'}}]:
        with pytest.raises(ValueError):
            check_convergence_acceptance(payload, bad, hashes, 'metrics')
    with pytest.raises(ValueError, match='counting'):
        check_convergence_acceptance({**payload, 'counting_method': 'old'}, acceptance, hashes, 'metrics')


def test_converged_matrices_need_no_waiver():
    payload = {'counting_method': 'headerless_all_pair_occurrences_v2',
               'convergence': {'is_converged': True}}
    assert check_convergence_acceptance(payload, None, {}, 'metrics') == 'converged'
