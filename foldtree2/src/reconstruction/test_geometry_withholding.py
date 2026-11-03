"""Self-checks for the geometry-withholding path (CPU, tiny model).

Run with:  python -m pytest foldtree2/src/reconstruction/test_geometry_withholding.py
       or:  python foldtree2/src/reconstruction/test_geometry_withholding.py
"""

import torch

from foldtree2.src.reconstruction.msa_transformer_adapter import MsaReconstructionHead
from foldtree2.src.reconstruction.train_msa_reconstruction import MsaReconstructionModule


def test_withhold_ratio_bounds():
    known = torch.tensor([[True, True, False, True]])

    m = MsaReconstructionModule(geometry_withhold_ratio=0.0)
    assert torch.equal(m._geometry_known(known), known)

    m = MsaReconstructionModule(geometry_withhold_ratio=1.0)
    assert m._geometry_known(known).sum().item() == 0

    m = MsaReconstructionModule(geometry_withhold_ratio=0.5)
    out = m._geometry_known(known)
    # Never reveals geometry for a row that has none, and never keeps all of it.
    assert (out & ~known).sum().item() == 0
    assert out.sum().item() <= 3


def test_marker_distinguishes_withheld_rows():
    torch.manual_seed(0)
    enc = MsaReconstructionHead(
        alphabet_size=31, embed_dim=32, num_heads=4, num_layers=1,
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, geom_dim=9,
    )
    enc.eval()
    tokens = torch.randint(1, 31, (1, 3, 8))
    geom = torch.zeros(1, 3, 8, 9)  # identical geometry for both branches
    visible = torch.tensor([[True, True, True]])
    hidden = torch.tensor([[False, False, False]])

    a = enc(tokens, geom_features=geom, geom_known_mask=visible)["features"]
    b = enc(tokens, geom_features=geom, geom_known_mask=hidden)["features"]
    assert a.shape == b.shape
    # Geometry is zero in both cases, so the only signal is the learnable
    # "no geometry" marker; the two runs must therefore differ.
    assert (a - b).abs().max().item() > 1e-6

    # Omitting the mask means "everything visible" (backwards compatible).
    c = enc(tokens, geom_features=geom)["features"]
    assert (a - c).abs().max().item() < 1e-6


def test_bad_mask_shape_raises():
    enc = MsaReconstructionHead(alphabet_size=31, embed_dim=32, num_heads=4,
                                num_layers=1, geom_dim=9)
    tokens = torch.randint(1, 31, (1, 3, 8))
    geom = torch.zeros(1, 3, 8, 9)
    try:
        enc(tokens, geom_features=geom, geom_known_mask=torch.ones(3, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("geom_known_mask with the wrong shape should raise")


def test_propagated_distances_exclude_self():
    m = MsaReconstructionModule(geometry_withhold_ratio=0.0, propagate_geometry=True)
    coords = torch.tensor([[
        [[0., 0, 0], [10., 0, 0], [20., 0, 0]],   # row 0: has structure
        [[0., 0, 0], [4., 0, 0], [8., 0, 0]],     # row 1: has structure
        [[0., 0, 0], [0., 0, 0], [0., 0, 0]],     # row 2: no structure
    ]])
    known = torch.tensor([[True, True, False]])
    pairs = torch.tensor([[0, 2]])
    prop = m._propagated_distances(coords, known, pairs)
    # A row never receives its own distance; it gets the mean of its homologs.
    assert abs(prop[0, 0, 0].item() - 8.0) < 1e-6
    assert abs(prop[0, 1, 0].item() - 20.0) < 1e-6
    assert abs(prop[0, 2, 0].item() - 14.0) < 1e-6


def test_ft2_alphabet_mapping():
    from foldtree2.src.reconstruction.tree_msa_dataset import ft2_token_map, tokenise
    tm = ft2_token_map(50)
    # RAxML-unsafe states are stored as 0xf7-0xff: '#'(35)->chr(247), '\n'(10)->chr(255)
    assert tm[ord(chr(247))] == 35
    assert tm[ord(chr(255))] == 10
    assert tm[1] == 1 and tm[40] == 40
    assert tokenise("-" + chr(247) + chr(255), tm).tolist() == [0, 35, 10]


def test_mafft_sentinel_stripped():
    from foldtree2.src.reconstruction.tree_msa_dataset import _strip_mafft_sentinel
    assert _strip_mafft_sentinel(chr(3) + chr(255) + "---") == chr(3) + "---"
    assert _strip_mafft_sentinel(chr(3) + chr(4) + "---") == chr(3) + chr(4) + "---"


if __name__ == "__main__":
    test_withhold_ratio_bounds()
    test_marker_distinguishes_withheld_rows()
    test_bad_mask_shape_raises()
    test_propagated_distances_exclude_self()
    test_ft2_alphabet_mapping()
    test_mafft_sentinel_stripped()
    print("geometry-withholding self-checks passed")
