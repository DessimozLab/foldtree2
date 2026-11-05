"""Side-chain orientation probe: how well do homolog side chains transfer?

Run with:  python -m foldtree2.src.reconstruction.sidechain_orientation_probe

Every PDB atom is rigid-fit into the family frame using the same transform the
dataset applies to CA, so donor and target coordinates are comparable.
For each column we already have the CB estimate (pure backbone-frame error).
For every further side-chain atom we report atom_err - cb_err, which is the
orientation error of the side chain itself.
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from foldtree2.src.reconstruction.tree_msa_dataset import FamilyMsaSample, procrustes


def frame(n, ca, c):
    e1 = ca - c
    n1 = np.linalg.norm(e1)
    if n1 < 1e-6:
        return None
    e1 = e1 / n1
    v = n - ca
    e2 = v - e1 * np.dot(v, e1)
    n2 = np.linalg.norm(e2)
    if n2 < 1e-6:
        return None
    e2 = e2 / n2
    return e1, e2, np.cross(e1, e2)


def parse_residues(pdb_path):
    order, cur = [], None
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            resnum = int(line[22:27])
        except ValueError:
            continue
        if cur is None or cur[0] != resnum:
            cur = (resnum, line[17:20].strip(), {})
            order.append(cur)
        cur[2][line[12:16].strip()] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return [(name, atoms) for _, name, atoms in order]


def fit_atoms(fam, taxon, bb, conf):
    """All PDB atoms of `taxon`, rigid-transformed into the family frame."""
    raw = parse_residues(fam / "structs" / f"{taxon}.pdb")
    if len(raw) != len(bb):
        return None
    cols = list(conf[taxon].items())
    if len(cols) < 30:
        return None
    P = np.array([bb[k][1] for _, k in cols])
    Q = np.array([raw[k][1].get("CA") for _, k in cols])
    if any(q is None for q in Q):
        return None
    r, t = procrustes(torch.tensor(P, dtype=torch.float32),
                     torch.tensor(Q, dtype=torch.float32))
    r, t = r.numpy(), t.numpy()
    return [(name, {a: v @ r.T + t for a, v in at.items()}) for name, at in raw]


def analyse(fam, thresh=70.0):
    s = FamilyMsaSample(fam, 51)
    prior = s._build_backbone_prior(thresh)
    conf = s._confident_columns(thresh)
    bb = {t: s._bb_coords[t].numpy() for t in s.taxon_ids}
    atoms = {}
    for t in s.taxon_ids:
        if (fam / "structs" / f"{t}.pdb").exists():
            a = fit_atoms(fam, t, bb[t], conf)
            if a is not None:
                atoms[t] = a

    rows = defaultdict(list)
    for t in s.taxon_ids:
        if t not in atoms:
            continue
        for c, k in conf[t].items():
            donors = [o for o in conf if o != t and c in conf[o]
                      and o in atoms and np.isfinite(bb[o][conf[o][c]]).all()
                      and atoms[o][conf[o][c]][0] == atoms[t][k][0]]
            if len(donors) < 2:
                continue
            Np = np.mean([bb[o][conf[o][c]][0] for o in donors], 0)
            CAp = np.mean([bb[o][conf[o][c]][1] for o in donors], 0)
            Cp = np.mean([bb[o][conf[o][c]][2] for o in donors], 0)
            f = frame(Np, CAp, Cp)
            if f is None:
                continue
            e1, e2, e3 = f
            target = atoms[t][k][1]
            if "CB" not in target:
                continue
            cb_err = float(np.linalg.norm(prior[t][c][4].numpy() - target["CB"]))
            for atom, pos in target.items():
                if atom in ("N", "CA", "C", "O", "CB"):
                    continue
                vecs = []
                for o in donors:
                    da = atoms[o][conf[o][c]][1]
                    if atom not in da:
                        continue
                    df = frame(bb[o][conf[o][c]][0], bb[o][conf[o][c]][1], bb[o][conf[o][c]][2])
                    if df is None:
                        continue
                    v = da[atom] - bb[o][conf[o][c]][1]
                    vecs.append([np.dot(v, df[0]), np.dot(v, df[1]), np.dot(v, df[2])])
                if not vecs:
                    continue
                w = np.mean(vecs, 0)
                pred = CAp + w[0] * e1 + w[1] * e2 + w[2] * e3
                rows[atom].append((float(np.linalg.norm(pred - pos)), cb_err,
                                   float(np.linalg.norm(pos - CAp))))
    return rows


if __name__ == "__main__":
    root = Path("families/Information_benchmark/marker_genes")
    agg = defaultdict(list)
    n = 0
    for fam in sorted(root.iterdir()):
        if not (fam / "encoded.ASCIIaln.txt").exists():
            continue
        for k, v in analyse(fam).items():
            agg[k] += v
        n += 1
        if n % 50 == 0:
            print(f"{n} families", flush=True)
    total = sum(len(v) for v in agg.values())
    print(f"\n{total:,} side-chain atoms over {n} families")
    print(f"{'atom':5s} {'n':>9s} {'err':>7s} {'CBerr':>7s} {'extra':>7s} {'|CA-atom|':>9s}")
    for k in sorted(agg, key=lambda a: -len(agg[a])):
        v = np.array(agg[k])
        print(f"{k:5s} {len(v):9d} {v[:,0].mean():7.2f} {v[:,1].mean():7.2f} "
              f"{v[:,0].mean()-v[:,1].mean():7.2f} {v[:,2].mean():9.2f}")
