import pandas as pd
import pytest

from foldtree2.src.encoder import load_encoded_fasta
from foldtree2.makesubmat import build_char_set


@pytest.mark.parametrize('ending', ['\n', ''])
def test_reader_preserves_control_tokens_and_final_record(tmp_path, ending):
    path = tmp_path / 'encoded.fa'
    # Do not strip tabs or vertical/form feed characters: these are tokens.
    path.write_bytes(('\n>first\r\n\x01\t\v\f\r\n\n>last\nÿø' + ending).encode())
    frame = load_encoded_fasta(path)
    assert frame.seq.to_dict() == {'first': '\x01\t\v\f', 'last': 'ÿø'}
    assert frame.seqlen.to_dict() == {'first': 4, 'last': 2}


@pytest.mark.parametrize('content', ['>dup\nA\n>dup\nB\n', 'A\n', '>\nA\n', '>empty\n'])
def test_reader_rejects_malformed_records(tmp_path, content):
    path = tmp_path / 'encoded.fa'
    path.write_text(content)
    with pytest.raises(ValueError):
        load_encoded_fasta(path)


def test_matrix_alphabet_guard_rejects_spurious_and_missing_states():
    frame = pd.DataFrame({'seq': ['\x01\x02']})
    assert len(build_char_set(frame, expected_size=2)[0]) == 2
    for bad in ['\x01\x02\n', '\x01']:
        with pytest.raises(ValueError, match='codebook'):
            build_char_set(pd.DataFrame({'seq': [bad]}), expected_size=2)
