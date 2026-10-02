"""Tests for the shared treebuilder/benchmark ancestral character workflow."""
from pathlib import Path

import pytest

from foldtree2.src.ancestral import ancestral_command, ancestral_states_to_fasta, run_ancestral


def test_fitted_command_preserves_paths_and_freezes_parameters():
    command = ancestral_command('alignment with spaces.fa', 'tree.nwk', 'fitted model',
                                'output prefix', threads=8, overwrite=True, freeze_fitted=True)
    assert command[command.index('--msa') + 1] == 'alignment with spaces.fa'
    assert command[command.index('--model') + 1] == 'fitted model'
    assert command[command.index('--opt-model') + 1] == 'off'
    assert command[command.index('--opt-branches') + 1] == 'off'
    assert '--ancestral' in command and '--redo' in command
    with pytest.raises(ValueError, match='positive'):
        ancestral_command('a', 't', 'm', 'p', threads=0)


def test_legacy_command_keeps_optimization_defaults():
    command = ancestral_command('a', 't', 'MULTI20_GTR{matrix}+I', 'prefix')
    assert '--opt-model' not in command and '--opt-branches' not in command


def test_prefix_outputs_and_failure_detection(tmp_path):
    prefix = tmp_path / 'different output prefix'
    def runner(command):
        for suffix in ('States', 'Probs', 'Tree'):
            Path(str(prefix) + '.raxml.ancestral' + suffix).touch()
    outputs = run_ancestral('unrelated_alignment.fa', 'tree', 'model', prefix, runner=runner)
    assert outputs['states'] == Path(str(prefix) + '.raxml.ancestralStates')
    with pytest.raises(FileNotFoundError):
        run_ancestral('a', 't', 'm', tmp_path / 'missing', runner=lambda command: None)


def test_consensus_export_retains_native_symbols(tmp_path):
    source = tmp_path / 'states'
    source.write_text('ASR1\t01AZ\nNode2\t10ZA\n')
    destination = ancestral_states_to_fasta(source)
    assert Path(destination).read_text() == '>ASR1\n01AZ\n>Node2\n10ZA\n'
    with pytest.raises(ValueError, match='overwrite'):
        ancestral_states_to_fasta(source, source)


@pytest.mark.parametrize('contents', ['', 'bad row\n', 'N\tAA\nN\tAA\n', 'N\tAA\nM\tA\n'])
def test_invalid_consensus_tables_rejected(tmp_path, contents):
    source = tmp_path / 'states'
    source.write_text(contents)
    with pytest.raises(ValueError):
        ancestral_states_to_fasta(source)
