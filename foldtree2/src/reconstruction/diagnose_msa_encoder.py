"""Does the MSA Transformer actually learn geometry?

Three independent probes, all on the encoder alone (no SE(3) step):

1. ATTENTION -> CONTACTS.  Row attention between residue positions i and j is
   the pair-specific signal the encoder produces.  If the encoder learned
   structural proximity, attention should separate contacting pairs (<= 8 A)
   from distant ones.  Compared against a sequence-separation baseline, which
   is the strongest purely one-dimensional predictor.

2. FEATURE PROBE.  Train a pair MLP on encoder features from the training
   families and test it on families the encoder never saw.  Generalisation
   across families is the only honest measure of learned geometry; a probe
   that only fits the families it was trained on is memorising.

3. GEOMETRY ABLATION.  Run the encoder on the same MSA with the geometry
   channels present and with them zeroed.  How much the features move tells
   us whether the structural input is used at all.

Usage:
    python -m foldtree2.src.reconstruction.diagnose_msa_encoder \
        --checkpoint models/msa_reconstruction/v1_base/msa_recon-epoch=006.ckpt
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from foldtree2.src.reconstruction.tree_msa_dataset import TreeMsaDataset, msa_collate
from foldtree2.src.reconstruction.train_msa_reconstruction import MsaReconstructionModule

CONTACT_CUTOFF = 8.0
MIN_SEP = 12


def _pairs_for(L, max_pairs, device):
    idx = torch.arange(L, device=device)
    sep = (idx[None] - idx[:, None]).abs()
    cand = (sep >= MIN_SEP).nonzero()
    if cand.shape[0] > max_pairs:
        cand = cand[torch.randperm(cand.shape[0], device=device)[:max_pairs]]
    return cand


def _auc(scores, positives):
    """AUC of a score separating positives from negatives, computed by rank."""
    order = torch.argsort(scores)
    n_pos = int(positives.sum().item())
    n_neg = int(positives.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(scores.numel(), device=scores.device)
    pos_ranks = ranks[positives].float()
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _f1(scores, positives, threshold):
    pred = scores >= threshold
    tp = int((pred & positives).sum().item())
    fp = int((pred & ~positives).sum().item())
    fn = int((~pred & positives).sum().item())
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r), p, r


def collect(model, families, device, max_pairs, with_geometry=True):
    """Return (features_i, features_j, attention, distances, contacts, separations)."""
    fi, fj, attns, dists, contacts, seps = [], [], [], [], [], []
    for fam in families:
        batch = msa_collate([fam])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        tokens = batch["tokens"]
        msa_mask = batch["mask"]
        coords = batch["coords"]
        known = batch["known_structure"]
        B, N, L, _ = coords.shape
        with torch.no_grad():
            enc = model.msa_encoder(
                tokens, msa_mask=msa_mask, need_head_weights=True,
                geom_features=model._geom_features(coords, known) if with_geometry else None,
                geom_known_mask=known if with_geometry else None,
            )
        features = enc["features"]                                  # (B, N, L, D)
        attn = enc["row_attentions"][:, -1].mean(dim=0)            # (B, L, L)
        pairs = _pairs_for(L, max_pairs, device)
        i, j = pairs[:, 0], pairs[:, 1]
        c = torch.nan_to_num(coords)
        d = torch.linalg.norm(c[:, :, i] - c[:, :, j], dim=-1)     # (B, N, P)
        valid = known.unsqueeze(-1).expand(-1, -1, pairs.shape[0])
        b_idx, n_idx, p_idx = valid.nonzero(as_tuple=True)
        fi.append(features[b_idx, n_idx, i[p_idx]])
        fj.append(features[b_idx, n_idx, j[p_idx]])
        attns.append(attn[b_idx, i[p_idx], j[p_idx]])
        dists.append(d[b_idx, n_idx, p_idx])
        contacts.append(d[b_idx, n_idx, p_idx] <= CONTACT_CUTOFF)
        seps.append((j[p_idx] - i[p_idx]).float())
    return (torch.cat(fi), torch.cat(fj), torch.cat(attns), torch.cat(dists),
            torch.cat(contacts), torch.cat(seps))


def probe(fi, fj, seps, labels, train_idx, test_idx, n_bins, use_features=True):
    """Pair MLP on (f_i, f_j, separation, dot product); report held-out accuracy."""
    D = fi.shape[-1]
    in_dim = 2 * D + 4 if use_features else 3
    mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(),
                        nn.Linear(256, n_bins)).to(fi.device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def inp(a, b, s):
        sep = torch.stack([s / 1000.0, torch.log(s.clamp(min=1.0)) / math.log(1000.0),
                           (s >= 100).float()], dim=-1)
        if not use_features:
            return sep
        rel = ((a * b).sum(-1) / D).unsqueeze(-1)
        return torch.cat([a, b, sep, rel], dim=-1)

    x = inp(fi[train_idx], fj[train_idx], seps[train_idx])
    y = labels[train_idx]
    for _ in range(40):
        loss = loss_fn(mlp(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        pred = mlp(inp(fi[test_idx], fj[test_idx], seps[test_idx])).argmax(-1)
    lab = labels[test_idx]
    acc = float((pred == lab).float().mean())
    recalls = [float(((pred == lab) & (lab == b)).sum() / (lab == b).sum())
               for b in range(n_bins) if (lab == b).any()]
    return acc, float(torch.tensor(recalls).mean()), recalls


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/msa_reconstruction/v1_base/msa_recon-epoch=006.ckpt")
    p.add_argument("--families-root", default="families/Information_benchmark/marker_genes")
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--max-pairs", type=int, default=8000)
    p.add_argument("--max-len", type=int, default=1200,
                   help="Skip families whose alignment is longer than this "
                        "(attention over very long alignments blows up memory).")
    p.add_argument("--random-encoder", action="store_true",
                   help="Use an untrained (randomly initialised) encoder instead of "
                        "the checkpoint. If a random encoder's features predict "
                        "distances as well as the trained one, the probe is reading "
                        "trivial sequence/position information, not geometry.")
    p.add_argument("--ablate-geometry", action="store_true",
                   help="Run every probe with the geometry channels removed, so "
                        "the encoder sees sequence only. Without this the test "
                        "rows' own structure is injected and the probe measures "
                        "information present, not information inferred.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if args.random_encoder:
        hp = {"alphabet_size": 51, "embed_dim": 128, "num_heads": 4, "num_layers": 4,
              "geometry_withhold_ratio": 0.3}
        embed_dim = hp["embed_dim"]
    else:
        ck = torch.load(ckpt, map_location=device)
        hp = ck.get("hyper_parameters", {})
        embed_dim = int(hp.get("embed_dim") or
                        ck["state_dict"]["msa_encoder.embed_tokens.weight"].shape[1])
    model = MsaReconstructionModule(
        alphabet_size=hp.get("alphabet_size", 51),
        embed_dim=embed_dim,
        num_heads=hp.get("num_heads", 4),
        num_layers=hp.get("num_layers", 4),
        geometry_withhold_ratio=hp.get("geometry_withhold_ratio", 0.3),
        max_pairs=args.max_pairs,
        loss_weights={"distance": 1.0},
    )
    if not args.random_encoder:
        state = ck["state_dict"]
        for k, v in model.state_dict().items():
            state.setdefault(k, v)
        model.load_state_dict(state)
    model.to(device).eval()
    print(f"encoder: {'random init' if args.random_encoder else ckpt.name}  "
          f"embed_dim={embed_dim} layers={hp.get('num_layers', 4)}")

    ds = TreeMsaDataset(families_root=args.families_root, alphabet_size=hp.get("alphabet_size", 51))
    all_fams = [i for i in range(ds.n_families)
                if max(len(ds._samples[i]._sequences[t])
                       for t in ds._samples[i].taxon_ids) <= args.max_len]
    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(all_fams), generator=rng).tolist()
    train_fams = [ds[i] for i in order[:args.n_train]]
    test_fams = [ds[i] for i in order[args.n_train:args.n_train + args.n_test]]
    print(f"{len(all_fams)} families with alignment length <= {args.max_len}")

    print(f"\ntrain families {len(train_fams)}  test families {len(test_fams)}")

    # ---------------- 1. attention -> contacts ---------------- #
    fi, fj, attn, dist, contact, sep = collect(model, test_fams, device, args.max_pairs,
                                               with_geometry=not args.ablate_geometry)
    n_pairs = int(dist.numel())
    n_contact = int(contact.sum().item())
    print("\n=== 1. row attention vs true contacts ===")
    print(f"  pairs {n_pairs}, contacts (<=8 A) {n_contact} ({100*n_contact/n_pairs:.1f}%)")
    print(f"  AUC  attention      : {_auc(attn, contact):.4f}")
    print(f"  AUC  separation     : {_auc(-sep, contact):.4f}")
    print(f"  AUC  attention*sep  : {_auc(attn * sep.float(), contact):.4f}")
    thr = float(attn.quantile(0.99))
    f1, prec, rec = _f1(attn, contact, thr)
    print(f"  F1 at 99th pct attention threshold: {f1:.4f} (precision {prec:.3f}, recall {rec:.3f})")
    corr = float(torch.corrcoef(torch.stack([attn, dist]))[0, 1])
    print(f"  corr(attention, distance) = {corr:.4f}")

    # ---------------- 2. generalising feature probe ---------------- #
    fi_tr, fj_tr, _, dist_tr, _, sep_tr = collect(model, train_fams, device, args.max_pairs,
                                                  with_geometry=not args.ablate_geometry)
    lab_tr = (dist_tr.unsqueeze(-1) >= model.distance_head.edges).sum(-1)
    lab_te = (dist.unsqueeze(-1) >= model.distance_head.edges).sum(-1)
    print("\n=== 2. pair probe: trained on train families, tested on unseen families ===")
    fi_all = torch.cat([fi_tr, fi]); fj_all = torch.cat([fj_tr, fj])
    sep_all = torch.cat([sep_tr, sep]); lab_all = torch.cat([lab_tr, lab_te])
    idx_tr = torch.arange(len(fi_tr), device=device)
    idx_te = torch.arange(len(fi_tr), len(fi_all), device=device)
    acc, macro, rec = probe(fi_all, fj_all, sep_all, lab_all, idx_tr, idx_te, 16, use_features=True)
    acc0, macro0, rec0 = probe(fi_all, fj_all, sep_all, lab_all, idx_tr, idx_te, 16, use_features=False)
    majority = float(torch.bincount(lab_te, minlength=16).max() / len(lab_te))
    counts = [int((lab_te == b).sum()) for b in range(16)]
    print(f"  held-out families   : features+separation acc {acc:.3f} macro {macro:.4f}")
    print(f"  held-out families   : separation only     acc {acc0:.3f} macro {macro0:.4f}")
    print(f"  held-out majority   : {majority:.3f}")
    print(f"  held-out bin counts : {counts}")
    print(f"  per-bin recall      : {[round(r, 3) for r in rec]}")

    # ---------------- 3. geometry ablation ---------------- #
    print("\n=== 3. does the encoder use the geometry channels? ===")
    fam = test_fams[0]
    batch = msa_collate([fam])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    coords, known = batch["coords"], batch["known_structure"]
    with torch.no_grad():
        geom = model._geom_features(coords, known)
        f_with = model.msa_encoder(batch["tokens"], msa_mask=batch["mask"],
                                   geom_features=geom, geom_known_mask=known)["features"]
        f_without = model.msa_encoder(batch["tokens"], msa_mask=batch["mask"])["features"]
    delta = (f_with - f_without).norm(dim=-1)
    base = f_without.norm(dim=-1)
    print(f"  family {fam['family_id']}: mean |feature change| {delta.mean():.3f} "
          f"vs mean feature norm {base.mean():.3f}  ({100*delta.mean()/base.mean():.1f}%)")
    geom_only = model.msa_encoder(batch["tokens"], msa_mask=batch["mask"],
                                  geom_features=geom, geom_known_mask=torch.zeros_like(known))["features"]
    d2 = (geom_only - f_without).norm(dim=-1)
    print(f"  zeroing geometry but keeping the 'no geometry' marker changes features by "
          f"{d2.mean():.3f}")


if __name__ == "__main__":
    main()
