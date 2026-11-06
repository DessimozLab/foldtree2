"""Self-checks for ancestral ft2 states, their decoded amino acids, and how
the geometry dataset consumes them.

Run:  python -m pytest foldtree2/src/reconstruction/test_ancestral_states.py
"""
from pathlib import Path
from types import SimpleNamespace

import torch

from foldtree2.src.reconstruction.ancestral_states import (
    ANCESTRAL_AA, ANCESTRAL_FT2, decode, decoder_ids, gap_mask, gapped,
    reconstruct, raxml_symbols, raxml_to_ft2, write_raxml_alignment)
from foldtree2.src.reconstruction.tree_msa_dataset import (
    FamilyMsaSample, read_fasta, _strip_mafft_sentinel)

FAMILY = Path("families/Information_benchmark/marker_genes/1003993")
ENCODER = Path("models/production/40char_minimal_decoder/"
               "final_40char_mk2_contactsfix_aa_encoder_full_epoch_40.pt")
DECODER = Path("models/production/40char_minimal_decoder/"
               "final_40char_mk2_contactsfix_aa_decoder_full_epoch_40.pt")
RAXML = Path("/home/dmoi/miniforge3/envs/foldtree2/bin/raxml-ng")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _stub_builder():
    """The attributes the mapping helpers need from a treebuilder."""
    replace = {chr(0): chr(246), '"': chr(248), '#': chr(247), '>': chr(249),
               '=': chr(250), '<': chr(251), '-': chr(252), ' ': chr(253),
               '\r': chr(254), '\n': chr(255)}
    alphabet = [replace.get(chr(c + 1), chr(c + 1)) for c in range(40)]
    return SimpleNamespace(alphabet=alphabet,
                           rev_replace_dict={v: k for k, v in replace.items()},
                           map={c: i for i, c in enumerate(sorted(alphabet))},
                           raxmlchars=list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "!\"#$%&'()*+,/:;<=>@[\\]^_{|}~"))


def test_gap_mask_keeps_only_columns_with_residues():
    # Node covers columns 0,1,2; its only leaf has a residue in 0 and 2.
    ancestral = "ABC"
    leaves = ["A-B"]
    mask = gap_mask(ancestral, leaves)
    assert mask == [True, False, True], mask
    assert gapped(ancestral, mask) == "A-C"


def test_raxml_alignment_round_trip():
    builder = _stub_builder()
    symbols = raxml_symbols(builder)
    back = raxml_to_ft2(builder)
    sequences = {"a": chr(1) + "-" + chr(3), "b": chr(2) + chr(4) + "-"}
    out = Path("/tmp/opencode/round_trip.fasta")
    write_raxml_alignment(sequences, symbols, out)
    read_back = read_fasta(out)
    assert read_back["a"] == "0-2" and read_back["b"] == "13-"
    assert {k: "".join(back.get(c, "-") for c in v) for k, v in read_back.items()} == sequences
    # Gaps stay gaps: they are not mapped into the state alphabet.
    assert "-" not in builder.alphabet


def test_decoder_ids_use_the_unreplaced_state_number():
    builder = _stub_builder()
    ids = decoder_ids(builder)
    symbols = raxml_symbols(builder)
    assert ids[symbols[chr(1)]] == 0 and ids[symbols[chr(40)]] == 39
    # MAFFT's sentinel chr(255) is the substituted form of chr(10), i.e. state 9.
    assert ids[symbols[chr(255)]] == 9


def test_ancestral_rows_match_the_alignment_layout():
    if not (FAMILY.is_dir() and ENCODER.is_file() and DECODER.is_file()
            and RAXML.is_file()):
        raise Exception("fixture family or models missing")
    outdir = Path("/tmp/opencode/ancestral_test")
    ft2 = reconstruct(FAMILY, ENCODER, DECODER, raxml_ng=str(RAXML),
                      outdir=outdir, overwrite=True)
    aa = read_fasta(outdir / ANCESTRAL_AA)
    alignment = {t: _strip_mafft_sentinel(s)
                 for t, s in read_fasta(
                     next(FAMILY.glob("*40char*/*encoded.ASCIIaln.txt"))).items()}
    length = max(len(s) for s in alignment.values())

    assert ft2, "no internal nodes reconstructed"
    assert set(ft2) == set(aa)
    for node, seq in ft2.items():
        assert len(seq) == length, f"{node}: {len(seq)} != {length}"
        # A node may only be a residue where a descendant leaf is one.
        for i, ch in enumerate(seq):
            if ch != "-":
                assert any(s[i] != "-" for s in alignment.values())
        # The decoded amino acids are defined exactly where the states are.
        assert len(aa[node]) == length
        assert (aa[node].count("-") == seq.count("-"))
        assert all(c in AMINO_ACIDS for c, b in zip(aa[node], seq) if b != "-")


def test_dataset_reads_ancestral_rows():
    outdir = Path("/tmp/opencode/ancestral_test")
    if not (outdir / ANCESTRAL_FT2).is_file():
        raise Exception("run test_ancestral_rows_match_the_alignment_layout first")
    sample = FamilyMsaSample(FAMILY, ancestral_files=(outdir / ANCESTRAL_FT2,
                                                      outdir / ANCESTRAL_AA))
    assert sample.ancestor_ids, "ancestral rows were not loaded"
    batch = sample.to_tensors()
    rows = sample.rows()
    ancestral_idx = [i for i, r in enumerate(rows) if r.startswith("anc:")]
    assert batch["tokens"].shape[0] == len(rows)
    assert not batch["known_structure"][ancestral_idx].any(), \
        "ancestral rows have no structure and must not be supervised"
    covered = batch["mask"][ancestral_idx]
    prior = batch["transfer_coords"][ancestral_idx]
    has_prior = torch.isfinite(prior).any(dim=-1)
    fraction = has_prior.sum().item() / covered.sum().item()
    assert fraction > 0.9, f"only {fraction:.2f} of ancestral columns have a prior"
    # The prior sits in the family frame, so at a column the reference structure
    # also covers it must be near that structure's CA, not in some arbitrary frame.
    ref = sample._frame_ref
    ref_cols = sample._confident_columns(sample.plddt_threshold)[ref]
    node = prior[0]
    shared = [c for c in ref_cols if torch.isfinite(node[c]).all()]
    assert len(shared) > 50, f"only {len(shared)} columns shared with the frame reference"
    ref_ca = sample._ca_coords[ref][[ref_cols[c] for c in shared]]
    distances = (node[shared] - ref_ca).norm(dim=1)
    assert distances.mean().item() < 8.0, f"prior {distances.mean():.1f} A from the frame"
