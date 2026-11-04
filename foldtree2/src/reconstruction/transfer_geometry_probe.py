#!/usr/bin/env python
"""Can geometry be TRANSFERRED from homologs instead of inferred from sequence?

The MSA Transformer cannot learn structure from these MSAs (3-20 sequences per
family - far too shallow for co-evolutionary signal).  But the geometry of an
ancestral or unstructured taxon is literally present in its homologs: if two
residues sit in the same alignment column they are, in principle, structurally
equivalent.

The catch is that every PDB file sits in its own arbitrary frame, so "the same
column is 35-260 A apart" is a trivially true statement about coordinates, not
about structure.  This probe superposes each taxon onto a reference with a
rigid Procrustes fit over the shared alignment columns, then measures:

  1. superposition RMSD per taxon  -> are alignment columns structurally
     equivalent at all?
  2. leave-one-out coordinate transfer: predict a taxon's coordinates as the
     mean of its superposed homologs at the same columns, and compare with the
     taxon's own (superposed) coordinates.

If (2) is small, the model does not need to infer geometry - it needs to learn
which homolog columns correspond to each residue.

Usage:
    python -m foldtree2.src.reconstruction.transfer_geometry_probe \
        --n-families 30 --plddt-threshold 70
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np


def procrustes(p: np.ndarray, q: np.ndarray):
    """Rigid transform (R, t) mapping q onto p, minimising RMSD (Kabsch)."""
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    h = (p - pc).T @ (q - qc)                      # 3x3 cross-covariance
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:                       # reflection guard
        u[:, -1] *= -1
        r = u @ vt
    return r, pc - r @ qc


def columns(seq: str, coords: np.ndarray, plddt: np.ndarray, thresh: float):
    """(column -> residue index) for confident, non-gap positions."""
    out = {}
    k = 0
    for c, ch in enumerate(seq):
        if ch != "-":
            if k < len(coords) and plddt[k] >= thresh:
                out[c] = k
            k += 1
    return out


def analyse(family, plddt_thresh: float):
    taxa = family.taxon_ids
    seqs = {t: family._sequences[t] for t in taxa}
    coords = {t: family._ca_coords[t].numpy() for t in taxa}
    plddt = {t: family._ca_plddt[t].numpy() for t in taxa}

    cols = {t: columns(seqs[t], coords[t], plddt[t], plddt_thresh) for t in taxa}
    ref = max(taxa, key=lambda t: len(cols[t]))
    if len(cols[ref]) < 30 or len(taxa) < 3:
        return None

    # Superpose every taxon onto the reference frame, keyed by alignment column.
    superposed, rmsd = {}, {}
    for t in taxa:
        shared = sorted(set(cols[ref]) & set(cols[t]))
        if len(shared) < 30:
            return None
        p = np.array([coords[ref][cols[ref][c]] for c in shared])
        q = np.array([coords[t][cols[t][c]] for c in shared])
        r, tr = procrustes(p, q)
        superposed[t] = {c: q[i] @ r.T + tr for i, c in enumerate(shared)}
        rmsd[t] = float(np.sqrt(((np.array([superposed[t][c] for c in shared]) - p)
                                ** 2).sum(axis=1).mean()))

    # Leave-one-out transfer: predict each taxon from its homologs.
    transfer, naive = [], []
    for h in taxa:
        donors = [t for t in taxa if t != h]
        pred, truth = [], []
        for c in cols[h]:
            vals = [superposed[t][c] for t in donors if c in superposed[t]]
            if vals:
                pred.append(np.mean(vals, axis=0))
                truth.append(superposed[h][c])
        if pred:
            transfer.append(float(np.sqrt(((np.array(pred) - np.array(truth))
                                          ** 2).sum(axis=1).mean())))
            d = donors[0]
            idx = [c for c in cols[h] if c in superposed[d]]
            if idx:
                naive.append(float(np.sqrt(((np.array([superposed[d][c] for c in idx])
                                            - np.array([superposed[h][c] for c in idx]))
                                           ** 2).sum(axis=1).mean())))

    return rmsd, transfer, naive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-root", default="families/Information_benchmark/marker_genes")
    ap.add_argument("--n-families", type=int, default=30)
    ap.add_argument("--plddt-threshold", type=float, default=70.0)
    args = ap.parse_args()

    from foldtree2.src.reconstruction.tree_msa_dataset import FamilyMsaSample

    fams = sorted(Path(args.families_root).iterdir())
    all_rmsd, all_transfer, all_naive, ok = [], [], [], 0
    for fam in fams:
        if not (fam / "encoded.ASCIIaln.txt").exists():
            continue
        try:
            sample = FamilyMsaSample(fam, alphabet_size=51, require_all_coords=False)
        except Exception:
            continue
        res = analyse(sample, args.plddt_threshold)
        if res is None:
            continue
        rmsd, transfer, naive = res
        all_rmsd.extend(rmsd.values())
        all_transfer.extend(transfer)
        all_naive.extend(naive)
        ok += 1
        if ok <= 6:
            print(f"{fam.name}: {len(sample.taxon_ids)} taxa, "
                  f"superposition RMSD mean {statistics.mean(rmsd.values()):.2f} A, "
                  f"transfer RMSD mean {statistics.mean(transfer):.2f} A "
                  f"(single homolog {statistics.mean(naive):.2f} A)")
        if ok >= args.n_families:
            break

    print(f"\nfamilies analysed: {ok}")
    print(f"superposition RMSD  : mean {statistics.mean(all_rmsd):.2f} A  "
          f"median {statistics.median(all_rmsd):.2f} A")
    print(f"leave-one-out transfer RMSD: mean {statistics.mean(all_transfer):.2f} A  "
          f"median {statistics.median(all_transfer):.2f} A")
    print(f"single-homolog transfer RMSD: mean {statistics.mean(all_naive):.2f} A  "
          f"median {statistics.median(all_naive):.2f} A")


if __name__ == "__main__":
    main()
