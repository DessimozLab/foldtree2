"""Dataset that loads OMA family MSAs of ft2 sequences with known PDB coordinates.

Each OMA family provides:
- Pre-encoded ft2 sequences (``encoded.fasta`` or ``encoded.ASCIIaln.txt``)
- A phylogenetic tree (``raxml_lg_tree.raxml.bestTree``)
- Per-taxon PDB structures under ``structs/``

During training a fraction of leaf sequences are **masked** from the loss
so the model must infer their geometry from the unmasked sequences —
simulating ancestral prediction at inference time.  CA coordinates from
the PDB files are converted to distance bin labels for supervision.
The ft2 alignment may contain gaps (MAFFT insertions); gap positions have
no PDB counterpart and are skipped when computing distance labels.
"""

from __future__ import annotations

import functools
import json
import math
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import torch
from torch.utils.data import Dataset

from foldtree2.src.distance.distance_restraints import (
    DISTANCE_EDGES,
    distance_labels as _distance_labels,
    eligible_pairs as _eligible_pairs,
)

# Gap/insertion token (index 0 is reserved for padding/gap).
_FT2_GAP = 0
# Minimum residue separation for supervised pairs.
_MIN_SEP = 12


# ---------------------------------------------------------------------------
# PDB coordinate extraction
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def ca_coordinates_from_pdb(pdb_path: str) -> torch.Tensor:
    """Extract CA coordinates from a PDB file using Gemmi.

    Returns a ``(N, 3)`` float32 tensor ordered by polymer-chain residue
    index.  Only standard amino acids are kept; ligands and water are
    skipped.  The best-model/best-chain heuristic picks the longest
    polymer chain (most CA atoms).  Result is cached.
    """
    if not Path(pdb_path).is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    st = gemmi.read_structure(pdb_path)
    best_model = None
    best_count = 0
    for model in st:
        count = sum(
            1 for chain in model for res in chain
            if not res.is_water()
            and gemmi.find_tabulated_residue(res.name).kind == gemmi.ResidueKind.AA
            and res.get_ca() is not None
        )
        if count > best_count:
            best_count = count
            best_model = model
    if best_model is None:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    best_chain = None
    best_len = 0
    for chain in best_model:
        coords = []
        for res in chain:
            if res.is_water():
                continue
            if gemmi.find_tabulated_residue(res.name).kind != gemmi.ResidueKind.AA:
                continue
            ca = res.get_ca()
            if ca is not None:
                coords.append((ca.pos.x, ca.pos.y, ca.pos.z))
        if len(coords) > best_len:
            best_len = len(coords)
            best_chain = coords
    if best_chain is None or best_len == 0:
        raise ValueError(f"No CA atoms found in polymer chain of {pdb_path}")
    return torch.tensor(np.array(best_chain, dtype=np.float32))


# ---------------------------------------------------------------------------
# Sequence I/O helpers
# ---------------------------------------------------------------------------

def read_fasta(path: Path, encoding: str = "latin-1") -> Dict[str, str]:
    """Mapping header -> sequence (latin-1 for ft2 byte-range chars)."""
    records: Dict[str, str] = {}
    current = ""
    for line in path.read_text(encoding=encoding).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            records[current] = ""
        else:
            records[current] = records.get(current, "") + line
    return records


read_aligned_ascii = read_fasta


def tokenise(sequence: str, alphabet_size: int = 31) -> torch.LongTensor:
    """Convert ft2 chars to token indices.  Index 0 = gap/pad."""
    tokens = torch.zeros(len(sequence), dtype=torch.long)
    for i, char in enumerate(sequence):
        code = ord(char) - 1
        if 0 <= code < alphabet_size - 1:
            tokens[i] = code + 1
        else:
            tokens[i] = _FT2_GAP
    return tokens


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def _pad_tensors(key_tensors, max_n, max_l, pad_pair_dim=False):
    """Pad a list of per-sample tensors to max_n x max_l (and optionally max_p)."""
    arr = []
    for t in key_tensors:
        n_pad = max_n - t.shape[0]
        l_pad = max_l - t.shape[1] if t.ndim >= 2 and not pad_pair_dim else 0
        if pad_pair_dim and t.ndim >= 2:
            # For distance_labels: same n_pad but needs pair-dim padding
            n_pad = max_n - t.shape[0]
            if n_pad > 0:
                pad = t.new_full((n_pad,) + t.shape[1:], -1, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            arr.append(t)
            continue
        if l_pad > 0:
            pad = torch.full((t.shape[0], l_pad) + t.shape[2:],
                             float("nan") if t.is_floating_point() else _FT2_GAP,
                             dtype=t.dtype)
            t = torch.cat([t, pad], dim=1)
        if n_pad > 0:
            pad = torch.full((n_pad,) + t.shape[1:],
                             float("nan") if t.is_floating_point() else _FT2_GAP,
                             dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        arr.append(t)
    return torch.stack(arr, dim=0)


def msa_collate(batch):
    """Collate variable-length MSAs into padded ``(B, max_N, max_L)`` tensors."""
    max_n_seq = max(s["tokens"].shape[0] for s in batch)
    max_len = max(s["tokens"].shape[1] for s in batch)

    out: dict = {"family_id": [s.get("family_id") for s in batch],
                  "index": [s.get("index") for s in batch],
                  "pairs": [s.get("pairs") for s in batch]}

    for key in ("tokens", "mask", "known_structure", "coords", "distance_labels"):
        tensors = [s[key] for s in batch]
        if key == "distance_labels":
            # Distance labels have variable P per family — pad to max P
            max_p = max(t.shape[-1] for t in tensors)
            arr = []
            for t in tensors:
                n_pad = max_n_seq - t.shape[0]
                if n_pad > 0:
                    t = torch.cat([t, t.new_full((n_pad, t.shape[1]), -1)], dim=0)
                if t.shape[1] < max_p:
                    t = torch.cat([t, t.new_full((t.shape[0], max_p - t.shape[1]), -1)], dim=1)
                arr.append(t)
            out[key] = torch.stack(arr, dim=0)
        elif tensors[0].ndim == 2:
            out[key] = _pad_tensors(tensors, max_n_seq, max_len)
        elif tensors[0].ndim == 3:
            out[key] = _pad_tensors(tensors, max_n_seq, max_len)
        elif tensors[0].ndim == 1:
            arr = []
            for t in tensors:
                n_pad = max_n_seq - t.shape[0]
                if n_pad > 0:
                    t = torch.cat([t, t.new_full((n_pad,), _FT2_GAP)], dim=0)
                arr.append(t)
            out[key] = torch.stack(arr, dim=0)

    out["padding_mask"] = out["tokens"] != _FT2_GAP
    return out


# ---------------------------------------------------------------------------
# One OMA family
# ---------------------------------------------------------------------------

class FamilyMsaSample:
    """Pre-processed ft2 MSA + CA coordinates for one OMA family.

    Args:
        family_dir: Path to the OMA family subdirectory.
        alphabet_size: ft2 vocab size (default 31: 30 chars + gap).
        require_all_coords: Raise if a taxon lacks a PDB.
        edges: Distance-bin boundaries (A).  Default ``DISTANCE_EDGES``.
    """

    def __init__(
        self,
        family_dir: Path,
        alphabet_size: int = 31,
        require_all_coords: bool = True,
        edges=DISTANCE_EDGES,
    ):
        self.family_dir = Path(family_dir)
        self.family_id = self.family_dir.name
        self.alphabet_size = int(alphabet_size)

        # --- ft2 aligned sequences ---
        aln_path = self.family_dir / "encoded.ASCIIaln.txt"
        if aln_path.is_file():
            ft2_seqs = read_aligned_ascii(aln_path)
        else:
            fasta_path = self.family_dir / "encoded.fasta"
            if fasta_path.is_file():
                ft2_seqs = read_fasta(fasta_path)
            else:
                raise FileNotFoundError(
                    f"No ft2 file in {family_dir}"
                )

        struct_dir = self.family_dir / "structs"
        if not struct_dir.is_dir():
            raise FileNotFoundError(f"No structs dir in {family_dir}")

        self._edges = torch.tensor(edges, dtype=torch.float32)

        # Match PDB taxa to ft2 headers
        self.taxon_ids: List[str] = sorted(
            p.stem for p in sorted(struct_dir.glob("*.pdb"))
            if p.stem in ft2_seqs
        )
        if not self.taxon_ids:
            raise ValueError(f"No ft2-aligned taxa with PDBs in {family_dir}")

        self._sequences: Dict[str, str] = {t: ft2_seqs[t] for t in self.taxon_ids}

        # --- Pre-load CA coords ---
        self._ca_coords: Dict[str, torch.Tensor] = {}
        for taxon in self.taxon_ids:
            pdb = struct_dir / f"{taxon}.pdb"
            if not pdb.is_file():
                if require_all_coords:
                    raise FileNotFoundError(f"Missing PDB for {taxon} in {family_dir}")
                continue
            self._ca_coords[taxon] = ca_coordinates_from_pdb(str(pdb))

        # --- Distance labels (shared pairs, per-taxon labels) ---
        self._distance_labels: Dict[str, torch.Tensor] = {}
        self._pairs: torch.Tensor = torch.empty(0, 2, dtype=torch.long)
        self._residue_mask: Dict[str, torch.Tensor] = {}
        self._shared_pairs_init = False

        for taxon in self.taxon_ids:
            coords = self._ca_coords.get(taxon)
            if coords is None:
                continue
            seq = self._sequences[taxon]
            tok = tokenise(seq, self.alphabet_size)
            valid = tok != _FT2_GAP
            n_valid = int(valid.sum().item())

            if coords.shape[0] != n_valid:
                warnings.warn(
                    f"{taxon}: {n_valid} non-gap vs "
                    f"{coords.shape[0]} CA; truncating"
                )
                n = min(n_valid, coords.shape[0])
                coords = coords[:n]
                self._ca_coords[taxon] = coords
                cum = valid.cumsum(dim=0)
                valid = valid.clone()
                valid &= cum <= n

            self._residue_mask[taxon] = valid

            if not self._shared_pairs_init:
                self._pairs = _eligible_pairs(valid, min_separation=_MIN_SEP)
                self._shared_pairs_init = True

        # Per-taxon labels from shared pairs
        for taxon in self.taxon_ids:
            if taxon not in self._residue_mask:
                continue
            coords = self._ca_coords.get(taxon)
            if coords is None:
                continue
            valid = self._residue_mask[taxon]
            n_valid = int(valid.sum().item())
            valid_idx = valid.nonzero(as_tuple=True)[0]
            ci = torch.searchsorted(valid_idx, self._pairs[:, 0])
            cj = torch.searchsorted(valid_idx, self._pairs[:, 1])
            # Pairs where either index maps outside valid range → invalid (-1)
            in_range = (ci < n_valid) & (cj < n_valid) & (ci != cj)
            labels = torch.full((self._pairs.shape[0],), -1, dtype=torch.long)
            if in_range.any():
                cp = torch.stack([ci[in_range], cj[in_range]], dim=1)
                good_labels, _ = _distance_labels(coords, cp, self._edges)
                labels[in_range] = good_labels.long()
            self._distance_labels[taxon] = labels

    def n_taxa(self) -> int:
        return len(self.taxon_ids)

    def to_tensors(
        self, mask_ratio: float = 0.0, rng: Optional[random.Random] = None
    ) -> dict:
        """Build training tensors.

        Returns:
            tokens:          (N_seq, L) long ft2 tokens
            mask:            (N_seq, L) bool — non-gap
            known_structure: (N_seq,) bool — supervised
            coords:          (N_seq, L, 3) — NaN at gaps/masked
            distance_labels: (N_seq, P) long, -1 = invalid
            pairs:           (P, 2) alignment-column-space
            family_id:       str
            n_taxa:          int
        """
        rng = rng or random
        taxa = list(self.taxon_ids)
        n_taxa = len(taxa)
        L = max(len(self._sequences[t]) for t in taxa)

        n_masked = max(1, int(n_taxa * mask_ratio)) if mask_ratio > 0 else 0
        masked_set = set(rng.sample(taxa, min(n_masked, n_taxa - 1))) if n_masked else set()

        tokens_list, mask_list = [], []
        known_list, coord_list, label_list = [], [], []
        n_pairs = self._pairs.shape[0]

        for taxon in taxa:
            seq = self._sequences.get(taxon, "")
            tok = tokenise(seq, self.alphabet_size)
            if len(tok) < L:
                tok = torch.cat([tok, tok.new_full((L - len(tok),), _FT2_GAP)])
            tokens_list.append(tok)
            valid = tok != _FT2_GAP
            mask_list.append(valid)

            is_known = taxon not in masked_set and taxon in self._residue_mask
            known_list.append(is_known)

            if is_known:
                ca = self._ca_coords[taxon].to(torch.float32)
                expanded = torch.full((L, 3), float("nan"), dtype=ca.dtype)
                rmask = self._residue_mask[taxon]
                if len(rmask) < L:
                    rmask = torch.cat([rmask, rmask.new_zeros(L - len(rmask), dtype=torch.bool)])
                n_rmask = int(rmask.sum().item())
                n_ca = ca.shape[0]
                if n_rmask != n_ca:
                    n = min(n_rmask, n_ca)
                    idx = rmask.nonzero(as_tuple=True)[0][:n]
                    expanded[idx] = ca[:n]
                else:
                    expanded[rmask] = ca
                coord_list.append(expanded)
                label_list.append(self._distance_labels[taxon])
            else:
                coord_list.append(torch.full((L, 3), float("nan"), dtype=torch.float32))
                label_list.append(torch.full((max(1, n_pairs),), -1, dtype=torch.long))

        return {
            "tokens": torch.stack(tokens_list, dim=0),
            "mask": torch.stack(mask_list, dim=0),
            "known_structure": torch.tensor(known_list, dtype=torch.bool),
            "coords": torch.stack(coord_list, dim=0),
            "distance_labels": torch.stack(label_list, dim=0),
            "pairs": self._pairs,
            "family_id": self.family_id,
            "n_taxa": n_taxa,
        }


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class TreeMsaDataset(Dataset):
    """Yields OMA family MSAs with CA coordinates for MSA reconstruction training."""

    def __init__(
        self,
        cohort_path: str = "configs/oma_benchmark_cohort.json",
        families_root: str = "families/Information_benchmark/marker_genes",
        mask_ratio: float = 0.25,
        family_ids: Optional[List[str]] = None,
        alphabet_size: int = 31,
    ):
        super().__init__()
        self.families_root = Path(families_root)
        self.mask_ratio = float(mask_ratio)
        self.alphabet_size = int(alphabet_size)
        self._rng = random.Random(42)

        cohort = json.loads(Path(cohort_path).read_text())
        families = (
            [f for f in cohort["families"] if f["family"] in family_ids]
            if family_ids else list(cohort["families"])
        )
        if not families:
            raise ValueError("No OMA families selected")

        self._samples: List[FamilyMsaSample] = []
        for entry in families:
            fam_dir = self.families_root / entry["family"]
            if not fam_dir.is_dir():
                continue
            try:
                sample = FamilyMsaSample(fam_dir, self.alphabet_size)
                if sample.n_taxa() >= 4:
                    self._samples.append(sample)
            except (FileNotFoundError, ValueError, RuntimeError):
                continue
        if not self._samples:
            raise RuntimeError("No valid OMA families with PDB structures found.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict:
        sample = self._samples[index]
        result = sample.to_tensors(mask_ratio=self.mask_ratio, rng=self._rng)
        result["index"] = index
        return result

    @property
    def n_families(self) -> int:
        return len(self._samples)

    @property
    def max_length(self) -> int:
        return max(len(s._sequences[t]) for s in self._samples
                   for t in s.taxon_ids)


class CladeSampledMsaDataset(Dataset):
    """Samples sub-clades from OMA family trees for varied MSAs."""

    def __init__(
        self,
        cohort_path: str = "configs/oma_benchmark_cohort.json",
        families_root: str = "families/Information_benchmark/marker_genes",
        clade_size: int = 8,
        samples_per_family: int = 5,
        mask_ratio: float = 0.25,
        alphabet_size: int = 31,
    ):
        super().__init__()
        self.families_root = Path(families_root)
        self.clade_size = int(clade_size)
        self.samples_per_family = int(samples_per_family)
        self.mask_ratio = float(mask_ratio)
        self.alphabet_size = int(alphabet_size)
        self._rng = random.Random(42)

        cohort = json.loads(Path(cohort_path).read_text())
        self._entries: List[Tuple[str, List[str]]] = []
        for entry in cohort["families"]:
            fam_id = entry["family"]
            taxa = [t["label"].split("|")[0] for t in entry["taxa"]
                    if len(t.get("label", "").split("|")) > 1]
            if len(taxa) < self.clade_size:
                self._entries.append((fam_id, taxa))
            else:
                for _ in range(self.samples_per_family):
                    self._entries.append((fam_id, self._rng.sample(taxa, self.clade_size)))

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict:
        fam_id, _ = self._entries[index]
        fam_dir = self.families_root / fam_id
        sample = FamilyMsaSample(fam_dir, self.alphabet_size)
        result = sample.to_tensors(mask_ratio=self.mask_ratio, rng=self._rng)
        result["index"] = index
        return result