#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

try:
    from numba import njit  # pyright: ignore[reportMissingImports]
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

PLDDT_ALPHABET   = "0123456789"  # 10 bins (project-standard pLDDT discretization)
CONTACT_ALPHABET = "01234567"    # 8 bins
RANGE_ALPHABET   = "LMG"         # local / mixed / global
BURIAL_ALPHABET  = "EIB"         # exposed / intermediate / buried
BIN_ALPHABET_BASE = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def read_structure_any(path: Path) -> gemmi.Structure:
    return gemmi.read_structure(str(path))


def structure_id_from_path(path: Path) -> str:
    name = path.name
    for suf in (".pdb.gz", ".cif.gz", ".mmcif.gz", ".pdb", ".cif", ".mmcif"):
        if name.endswith(suf):
            return name[:-len(suf)]
    return path.stem


def pick_model(structure: gemmi.Structure) -> gemmi.Model:
    if len(structure) == 0:
        raise ValueError("structure contains no models")
    return structure[0]


def is_polymer_residue(res: gemmi.Residue) -> bool:
    return res.name.upper() in AA3_TO_1


def get_atom_pos(res: gemmi.Residue, atom_name: str) -> Optional[np.ndarray]:
    atom_name = atom_name.strip()
    for atom in res:
        if atom.name.strip() == atom_name:
            p = atom.pos
            return np.array([p.x, p.y, p.z], dtype=np.float32)
    return None


def get_atom_bfactor(res: gemmi.Residue, atom_name: str) -> Optional[float]:
    atom_name = atom_name.strip()
    for atom in res:
        if atom.name.strip() == atom_name:
            return float(atom.b_iso)
    return None


def pseudo_cb_from_ca_n_c(ca: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    def unit(v: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(v)
        if nrm < 1e-8:
            return v
        return v / nrm

    b = unit(n - ca)
    cvec = unit(c - ca)
    a = unit(np.cross(b, cvec))
    direction = unit(-0.58273431 * b + -0.54067466 * cvec + 0.60704629 * a)
    return (ca + 1.522 * direction).astype(np.float32)


def extract_chain_arrays(chain: gemmi.Chain):
    aa_chars = []
    ca_list = []
    cb_list = []
    plddt_list = []

    for res in chain:
        if not is_polymer_residue(res):
            continue

        aa = AA3_TO_1.get(res.name.upper(), "X")
        if aa == "X":
            continue

        ca = get_atom_pos(res, "CA")
        if ca is None:
            continue

        cb = get_atom_pos(res, "CB")
        if cb is None:
            n = get_atom_pos(res, "N")
            c = get_atom_pos(res, "C")
            if n is not None and c is not None:
                cb = pseudo_cb_from_ca_n_c(ca, n, c)
            else:
                cb = ca

        plddt = get_atom_bfactor(res, "CA")
        if plddt is None:
            bfs = [float(atom.b_iso) for atom in res]
            plddt = float(np.mean(bfs)) if bfs else 0.0

        aa_chars.append(aa)
        ca_list.append(ca)
        cb_list.append(cb)
        plddt_list.append(plddt)

    if not aa_chars:
        return None

    return (
        "".join(aa_chars),
        np.asarray(ca_list, dtype=np.float32),
        np.asarray(cb_list, dtype=np.float32),
        np.asarray(plddt_list, dtype=np.float32),
    )


def extract_residue_tables(path: Path, chain_mode: str = "separate"):
    st = read_structure_any(path)
    model = pick_model(st)
    sid = structure_id_from_path(path)

    if chain_mode == "separate":
        out = []
        for chain in model:
            arr = extract_chain_arrays(chain)
            if arr is None:
                continue
            aa, ca_xyz, cb_xyz, plddt = arr
            out.append((f"{sid}|{chain.name}", aa, ca_xyz, cb_xyz, plddt))
        return out

    elif chain_mode == "merge":
        aa_parts = []
        ca_parts = []
        cb_parts = []
        plddt_parts = []
        for chain in model:
            arr = extract_chain_arrays(chain)
            if arr is None:
                continue
            aa, ca_xyz, cb_xyz, plddt = arr
            aa_parts.append(aa)
            ca_parts.append(ca_xyz)
            cb_parts.append(cb_xyz)
            plddt_parts.append(plddt)

        if not aa_parts:
            return []

        aa = "".join(aa_parts)
        ca_xyz = np.concatenate(ca_parts, axis=0)
        cb_xyz = np.concatenate(cb_parts, axis=0)
        plddt = np.concatenate(plddt_parts, axis=0)
        return [(sid, aa, ca_xyz, cb_xyz, plddt)]

    else:
        raise ValueError(f"unknown chain_mode={chain_mode}")


@njit(cache=True)
def _neighbor_stats_numba(neighbor_lens, neighbor_flat, local_seq_sep):
    L = neighbor_lens.shape[0]
    total = np.zeros(L, dtype=np.int16)
    local = np.zeros(L, dtype=np.int16)
    global_ = np.zeros(L, dtype=np.int16)

    start = 0
    for i in range(L):
        n = neighbor_lens[i]
        loc = 0
        tot = 0
        for k in range(start, start + n):
            j = neighbor_flat[k]
            if j == i:
                continue
            tot += 1
            if abs(j - i) <= local_seq_sep:
                loc += 1
        total[i] = tot
        local[i] = loc
        global_[i] = tot - loc
        start += n

    return total, local, global_


def compute_neighbor_tracks(
    xyz: np.ndarray,
    cutoff: float = 8.0,
    local_seq_sep: int = 8,
):
    L = xyz.shape[0]
    if L == 0:
        z = np.zeros(0, dtype=np.int16)
        return z, z, z

    tree = cKDTree(xyz)
    neigh = tree.query_ball_point(xyz, r=cutoff)

    if HAVE_NUMBA:
        lens = np.fromiter((len(x) for x in neigh), dtype=np.int32, count=L)
        total_len = int(lens.sum())
        flat = np.empty(total_len, dtype=np.int32)
        pos = 0
        for js in neigh:
            n = len(js)
            flat[pos:pos + n] = js
            pos += n
        return _neighbor_stats_numba(lens, flat, local_seq_sep)

    total = np.zeros(L, dtype=np.int16)
    local = np.zeros(L, dtype=np.int16)
    global_ = np.zeros(L, dtype=np.int16)

    for i, js in enumerate(neigh):
        if not js:
            continue
        arr = np.asarray(js, dtype=np.int32)
        arr = arr[arr != i]
        if arr.size == 0:
            continue

        dseq = np.abs(arr - i)
        loc = np.count_nonzero(dseq <= local_seq_sep)
        tot = arr.size

        total[i] = tot
        local[i] = loc
        global_[i] = tot - loc

    return total, local, global_


def make_range_bins(total: np.ndarray, global_: np.ndarray) -> np.ndarray:
    ratio = global_ / np.maximum(total, 1)
    out = np.zeros_like(total, dtype=np.int16)
    out[(ratio >= 0.33) & (ratio <= 0.66)] = 1
    out[ratio > 0.66] = 2
    return out


def make_burial_bins(total: np.ndarray) -> np.ndarray:
    """
    Cheap burial proxy from total contact count.
    3-state:
      0 = exposed
      1 = intermediate
      2 = buried
    Tune thresholds to dataset later.
    """
    out = np.zeros_like(total, dtype=np.int16)
    out[total >= 8] = 1
    out[total >= 16] = 2
    return out


def digitize_to_uint(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(values, bins=np.asarray(edges, dtype=np.float32), right=False).astype(np.uint8)


def plddt_to_discrete_bins(plddt: np.ndarray) -> np.ndarray:
    """Project-standard pLDDT binning: floor(plddt / 10), clipped to [0, 9]."""
    clipped = np.clip(plddt, 0.0, 100.0)
    return np.minimum((clipped // 10.0).astype(np.uint8), 9)


def make_bin_alphabet(n_bins: int) -> str:
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if n_bins > len(BIN_ALPHABET_BASE):
        raise ValueError(
            f"n_bins={n_bins} exceeds supported alphabet size={len(BIN_ALPHABET_BASE)}"
        )
    return BIN_ALPHABET_BASE[:n_bins]


def make_bend_bins(ca_xyz: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    CA bend angle bins per residue.
    Angle at i uses CA(i-1)-CA(i)-CA(i+1), then digitized to 8 bins over 0..180.
    Ends are padded from nearest valid interior value.
    """
    L = ca_xyz.shape[0]
    out = np.zeros(L, dtype=np.float32)
    if L < 3:
        return np.zeros(L, dtype=np.uint8)

    v1 = ca_xyz[:-2] - ca_xyz[1:-1]
    v2 = ca_xyz[2:] - ca_xyz[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = np.maximum(n1 * n2, 1e-8)
    cosang = np.sum(v1 * v2, axis=1) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang)).astype(np.float32)

    out[1:-1] = ang
    out[0] = ang[0]
    out[-1] = ang[-1]

    bend_edges = np.linspace(0.0, 180.0, n_bins + 1, dtype=np.float32)[1:-1].tolist()
    return digitize_to_uint(out, bend_edges)


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return 0.0
    b1u = b1 / b1_norm

    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm < 1e-8 or w_norm < 1e-8:
        return 0.0

    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def make_torsion_bins(ca_xyz: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """
    CA pseudo-dihedral bins per residue.
    Dihedral uses quadruplets CA(i-1),CA(i),CA(i+1),CA(i+2), assigned to residue i.
    Ends are padded from nearest valid interior value, then digitized to 8 bins over 360°.
    """
    L = ca_xyz.shape[0]
    out = np.zeros(L, dtype=np.float32)
    if L < 4:
        return np.zeros(L, dtype=np.uint8)

    for i in range(L - 3):
        out[i + 1] = _dihedral_deg(ca_xyz[i], ca_xyz[i + 1], ca_xyz[i + 2], ca_xyz[i + 3])

    out[0] = out[1]
    out[-2] = out[L - 3]
    out[-1] = out[L - 3]

    wrapped = (out + 180.0) % 360.0
    torsion_edges = np.linspace(0.0, 360.0, n_bins + 1, dtype=np.float32)[1:-1].tolist()
    return digitize_to_uint(wrapped, torsion_edges)


def uint_to_string(vals: np.ndarray, alphabet: str) -> str:
    lut = np.array(list(alphabet), dtype="<U1")
    return "".join(lut[vals].tolist())


def fasta_record(name: str, seq: str, width: int = 80) -> str:
    parts = [f">{name}\n"]
    for i in range(0, len(seq), width):
        parts.append(seq[i:i + width])
        parts.append("\n")
    return "".join(parts)


def process_one(
    path_str: str,
    cutoff: float,
    local_seq_sep: int,
    contact_edges: list[float],
    chain_mode: str,
    bend_n_bins: int,
    torsion_n_bins: int,
):
    path = Path(path_str)

    aa_chunks = []
    plddt_chunks = []
    contact_chunks = []
    range_chunks = []
    burial_chunks = []
    bend_chunks = []
    torsion_chunks = []
    bend_alphabet = make_bin_alphabet(bend_n_bins)
    torsion_alphabet = make_bin_alphabet(torsion_n_bins)

    try:
        tables = extract_residue_tables(path, chain_mode=chain_mode)

        for name, aa, ca_xyz, cb_xyz, plddt in tables:
            total, local, global_ = compute_neighbor_tracks(
                cb_xyz,
                cutoff=cutoff,
                local_seq_sep=local_seq_sep,
            )

            plddt_bins = plddt_to_discrete_bins(plddt)
            contact_bins = digitize_to_uint(total.astype(np.float32), contact_edges)
            range_bins = make_range_bins(total, global_).astype(np.uint8)
            burial_bins = make_burial_bins(total).astype(np.uint8)
            bend_bins = make_bend_bins(ca_xyz, n_bins=bend_n_bins)
            torsion_bins = make_torsion_bins(ca_xyz, n_bins=torsion_n_bins)

            aa_chunks.append(fasta_record(name, aa))
            plddt_chunks.append(fasta_record(name, uint_to_string(plddt_bins, PLDDT_ALPHABET)))
            contact_chunks.append(fasta_record(name, uint_to_string(contact_bins, CONTACT_ALPHABET)))
            range_chunks.append(fasta_record(name, uint_to_string(range_bins, RANGE_ALPHABET)))
            burial_chunks.append(fasta_record(name, uint_to_string(burial_bins, BURIAL_ALPHABET)))
            bend_chunks.append(fasta_record(name, uint_to_string(bend_bins, bend_alphabet)))
            torsion_chunks.append(fasta_record(name, uint_to_string(torsion_bins, torsion_alphabet)))

        return (
            "".join(aa_chunks),
            "".join(plddt_chunks),
            "".join(contact_chunks),
            "".join(range_chunks),
            "".join(burial_chunks),
            "".join(bend_chunks),
            "".join(torsion_chunks),
        )

    except Exception as e:
        print(f"[WARN] failed on {path}: {e}")
        return ("", "", "", "", "", "", "")


def process_one_star(args):
	return process_one(*args)


def iter_structure_files(indir: Path):
    valid = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz")
    for p in indir.rglob("*"):
        if p.is_file() and p.name.lower().endswith(valid):
            yield p


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--local-seq-sep", type=int, default=8)
    ap.add_argument("--chain-mode", choices=["separate", "merge"], default="separate")
    ap.add_argument("--bend-bins", type=int, default=8)
    ap.add_argument("--torsion-bins", type=int, default=12)
    ap.add_argument("--chunksize", type=int, default=16)
    args = ap.parse_args()
    if args.bend_bins < 2:
        raise SystemExit("--bend-bins must be >= 2")
    if args.torsion_bins < 2:
        raise SystemExit("--torsion-bins must be >= 2")
    if args.bend_bins > len(BIN_ALPHABET_BASE):
        raise SystemExit(f"--bend-bins exceeds supported max {len(BIN_ALPHABET_BASE)}")
    if args.torsion_bins > len(BIN_ALPHABET_BASE):
        raise SystemExit(f"--torsion-bins exceeds supported max {len(BIN_ALPHABET_BASE)}")
    return args


def main():
    args = parse_args()
    files = sorted(iter_structure_files(args.indir))
    if not files:
        raise SystemExit("No structure files found.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    aa_path = args.outdir / "aa.fasta"
    plddt_path = args.outdir / "plddt_bin.fasta"
    contact_path = args.outdir / "contact_bin.fasta"
    range_path = args.outdir / "range_bin.fasta"
    burial_path = args.outdir / "burial_bin.fasta"
    bend_path = args.outdir / "bend_bin.fasta"
    torsion_path = args.outdir / "torsion_bin.fasta"

    # len(edges)+1 must equal alphabet length
    contact_edges = [1, 3, 5, 8, 12, 16, 24]  # -> 8 bins

    worker_args = [
        (
            str(p),
            args.cutoff,
            args.local_seq_sep,
            contact_edges,
            args.chain_mode,
            args.bend_bins,
            args.torsion_bins,
        )
        for p in files
    ]

    with (
        open(aa_path, "w") as faa,
        open(plddt_path, "w") as fpl,
        open(contact_path, "w") as fco,
        open(range_path, "w") as frng,
        open(burial_path, "w") as fbu,
        open(bend_path, "w") as fbe,
        open(torsion_path, "w") as fto,
        mp.Pool(processes=args.workers) as pool,
    ):

        results = pool.imap(
            process_one_star,
            worker_args,
            chunksize=args.chunksize,
        )
        for aa_txt, plddt_txt, contact_txt, range_txt, burial_txt, bend_txt, torsion_txt in tqdm(
            results,
            total=len(worker_args),
            desc="Processing structures",
            unit="file",
        ):
            if aa_txt:
                faa.write(aa_txt)
                fpl.write(plddt_txt)
                fco.write(contact_txt)
                frng.write(range_txt)
                fbu.write(burial_txt)
                fbe.write(bend_txt)
                fto.write(torsion_txt)


if __name__ == "__main__":
    main()