import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from extend_matrix_references import select_members, concatenate_encoded_fastas


def test_selection_excludes_old_families_and_samples_members_reproducibly(tmp_path):
    table = tmp_path / 'clusters.tsv'
    table.write_text(''.join(f'{family}{i}\t{family}\t1\n'
                            for family, count in [('old', 8), ('new1', 10), ('new2', 12), ('small', 3)]
                            for i in range(count)))
    selected = select_members(table, {'old'}, max_families=10, members=5)
    assert set(selected) == {'new1', 'new2'}
    assert all(len(value) == 5 for value in selected.values())
    assert selected == select_members(table, {'old'}, max_families=10, members=5)


def test_concatenation_does_not_add_blank_lines_or_lose_last_record(tmp_path):
    from foldtree2.src.encoder import load_encoded_fasta
    first, second, output = [tmp_path / p for p in ['first.fa', 'second.fa', 'joined.fa']]
    first.write_bytes(b'>a\n\x01\t\n')
    second.write_bytes(b'>b\n\x02')
    concatenate_encoded_fastas([first, second], output)
    assert output.read_bytes() == b'>a\n\x01\t\n>b\n\x02\n'
    assert load_encoded_fasta(output).seq.to_dict() == {'a': '\x01\t', 'b': '\x02'}
