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


def test_oracle_distance_is_learnable():
    """Upper-bound control: with a metric in the input the head must be able to
    fit the bins.  If this passes but real training stays at chance, the encoder
    is the bottleneck, not the head or the loss."""
    from foldtree2.src.reconstruction.reconstruction_heads import MsaDistanceHead
    head = MsaDistanceHead(embed_dim=4, hidden=64, dropout=0.0, use_oracle=True)
    L = 30
    pairs = torch.tensor([[i, i + 12] for i in range(L - 12)])
    cycle = [3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]
    oracle = torch.tensor([cycle[k % len(cycle)] for k in range(pairs.shape[0])])
    oracle = oracle.reshape(1, 1, -1)
    labels = (oracle.unsqueeze(-1) >= head.edges).sum(-1)
    feats = torch.zeros(1, 1, L, 4)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(150):
        # chunk_size < P exercises the per-chunk slicing of the pair features.
        logits = head(feats, pairs=pairs, pair_oracle=oracle, chunk_size=7)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, head.n_bins),
                                                labels.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
    acc = (logits.argmax(-1) == labels).float().mean().item()
    assert acc > 0.9, f"head cannot fit oracle distances (acc {acc:.3f})"
    # A geometry-free row (oracle -1 -> 0) gets no metric signal at all.
    no_geom = head(feats, pairs=pairs, chunk_size=7,
                   pair_oracle=torch.full((1, 1, pairs.shape[0]), -1.0))
    pred = no_geom.argmax(-1)
    assert (pred[0, 0] == pred[0, 0, 0]).all()


def test_procrustes_recovers_rigid_transform():
    import math
    from foldtree2.src.reconstruction.tree_msa_dataset import procrustes
    torch.manual_seed(0)
    p = torch.randn(200, 3)
    c, s = math.cos(0.7), math.sin(0.7)
    r_true = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q = p @ r_true.T + torch.tensor([10.0, -3.0, 4.0])
    r, t = procrustes(p, q)
    back = q @ r.T + t
    assert torch.allclose(back, p, atol=1e-6), "Procrustes must invert the rotation"


def test_transfer_prior_excludes_self():
    from foldtree2.src.reconstruction.tree_msa_dataset import FamilyMsaSample
    s = FamilyMsaSample.__new__(FamilyMsaSample)
    s.taxon_ids = ["a", "b", "c"]
    s._sequences = {t: "AA" for t in s.taxon_ids}
    s._ca_coords = {
        "a": torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
        "b": torch.tensor([[1.0, 0.0, 0.0], [4.8, 0.0, 0.0]]),
        "c": torch.tensor([[3.0, 0.0, 0.0], [6.8, 0.0, 0.0]]),
    }
    s._ca_plddt = {t: torch.full((2,), 90.0) for t in s.taxon_ids}
    s._residue_mask = {t: torch.tensor([True, True]) for t in s.taxon_ids}
    prior = s._build_transfer_prior(70.0)
    # column 0 of 'a' is predicted from b and c only: (1 + 3) / 2 = 2
    assert torch.allclose(prior["a"][0], torch.tensor([2.0, 0.0, 0.0]))
    assert prior["a"].shape == (2, 3)


def test_prior_gap_fill():
    from foldtree2.src.reconstruction.train_msa_reconstruction import _fill_prior_gaps
    prior = torch.tensor([[0.0, 0, 0], [float("nan"), 0, 0],
                         [float("nan"), 0, 0], [4.0, 0, 0], [6.0, 0, 0]])
    filled = _fill_prior_gaps(prior)
    assert filled is not None
    # Positions 1 and 2 are nearest to residue 0 and residue 3 respectively.
    assert torch.allclose(filled[1], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(filled[2], torch.tensor([4.0, 0.0, 0.0]))
    assert torch.isfinite(filled).all(), "fill must remove every NaN"
    sparse = torch.tensor([[1.0, 0, 0], [float("nan"), 0, 0], [2.0, 0, 0]])
    assert _fill_prior_gaps(sparse) is None, "fewer than 3 covered residues is unusable"


if __name__ == "__main__":
    test_withhold_ratio_bounds()
    test_marker_distinguishes_withheld_rows()
    test_bad_mask_shape_raises()
    test_propagated_distances_exclude_self()
    test_ft2_alphabet_mapping()
    test_mafft_sentinel_stripped()
    test_oracle_distance_is_learnable()
    test_prior_gap_fill()
    print("geometry-withholding self-checks passed")
