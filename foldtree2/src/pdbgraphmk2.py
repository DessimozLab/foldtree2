import json
import multiprocessing as mp
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Optional

import gemmi
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch_geometric
import tqdm
from scipy import sparse
from scipy.spatial import cKDTree, distance
from torch_geometric.data import Dataset, HeteroData

from foldtree2.src.rigid_utils import rot_to_quat


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

BIN_ALPHABET_BASE = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Chi angle atom definitions for each residue type
# Each entry is a list of 4-atom tuples defining dihedral angles (chi1, chi2, chi3, chi4)
CHI_ATOMS = {
    "ALA": [],  # No chi angles
    "GLY": [],  # No chi angles
    "VAL": [["N", "CA", "CB", "CG1"]],  # chi1 only
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
}

# Maximum number of chi angles (for padding)
MAX_CHI = 4

# Sidechain heavy atoms for centroid calculation (excluding backbone N, CA, C, O)
SIDECHAIN_ATOMS = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],  # No sidechain
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}


def _atom_pos(res: gemmi.Residue, atom_name: str) -> Optional[np.ndarray]:
    atom_name = atom_name.strip()
    for atom in res:
        if atom.name.strip() == atom_name:
            p = atom.pos
            return np.array([p.x, p.y, p.z], dtype=np.float32)
    return None


def _atom_bfactor(res: gemmi.Residue, atom_name: str) -> Optional[float]:
    atom_name = atom_name.strip()
    for atom in res:
        if atom.name.strip() == atom_name:
            return float(atom.b_iso)
    return None


def _pseudo_cb_from_ca_n_c(ca: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    def _unit(v: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(v)
        if nrm < 1e-8:
            return v
        return v / nrm

    b = _unit(n - ca)
    cvec = _unit(c - ca)
    a = _unit(np.cross(b, cvec))
    direction = _unit(-0.58273431 * b + -0.54067466 * cvec + 0.60704629 * a)
    return (ca + 1.522 * direction).astype(np.float32)


def _digitize_to_uint(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(values, bins=np.asarray(edges, dtype=np.float32), right=False).astype(np.uint8)


def _make_bend_bins(ca_xyz: np.ndarray, n_bins: int = 8) -> np.ndarray:
    length = ca_xyz.shape[0]
    out = np.zeros(length, dtype=np.float32)
    if length < 3:
        return np.zeros(length, dtype=np.uint8)

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

    edges = np.linspace(0.0, 180.0, n_bins + 1, dtype=np.float32)[1:-1].tolist()
    return _digitize_to_uint(out, edges)


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


def _compute_chi_angles(res, res_name: str) -> np.ndarray:
    """
    Compute chi angles for a residue.
    
    Returns:
        np.ndarray: Shape (MAX_CHI * 2,) containing [sin(chi1), cos(chi1), sin(chi2), cos(chi2), ...].
                    Missing chi angles are filled with (0, 0).
    """
    chi_defs = CHI_ATOMS.get(res_name, [])
    result = np.zeros(MAX_CHI * 2, dtype=np.float32)
    
    for i, atom_names in enumerate(chi_defs):
        if i >= MAX_CHI:
            break
        
        # Get atom positions
        positions = []
        for atom_name in atom_names:
            atom = res.find_atom(atom_name, '\0')  # Gemmi API
            if atom is None:
                break
            positions.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
        
        if len(positions) == 4:
            angle_deg = _dihedral_deg(positions[0], positions[1], positions[2], positions[3])
            angle_rad = np.radians(angle_deg)
            result[i * 2] = np.sin(angle_rad)
            result[i * 2 + 1] = np.cos(angle_rad)
    
    return result


def _compute_sidechain_centroid(res, res_name: str, ca_pos: np.ndarray) -> np.ndarray:
    """
    Compute sidechain centroid direction and distance from CA.
    
    Returns:
        np.ndarray: Shape (4,) containing [dir_x, dir_y, dir_z, distance].
                    If no sidechain atoms, returns zeros.
    """
    sc_atoms = SIDECHAIN_ATOMS.get(res_name, [])
    result = np.zeros(4, dtype=np.float32)
    
    if not sc_atoms:
        return result
    
    positions = []
    for atom_name in sc_atoms:
        atom = res.find_atom(atom_name, '\0')
        if atom is not None:
            positions.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
    
    if not positions:
        return result
    
    centroid = np.mean(positions, axis=0)
    direction = centroid - ca_pos
    distance = np.linalg.norm(direction)
    
    if distance > 1e-6:
        result[:3] = direction / distance  # Normalized direction
        result[3] = distance
    
    return result


def _make_torsion_bins(ca_xyz: np.ndarray, n_bins: int = 12) -> np.ndarray:
    length = ca_xyz.shape[0]
    out = np.zeros(length, dtype=np.float32)
    if length < 4:
        return np.zeros(length, dtype=np.uint8)

    for i in range(length - 3):
        out[i + 1] = _dihedral_deg(ca_xyz[i], ca_xyz[i + 1], ca_xyz[i + 2], ca_xyz[i + 3])

    out[0] = out[1]
    out[-2] = out[length - 3]
    out[-1] = out[length - 3]

    wrapped = (out + 180.0) % 360.0
    edges = np.linspace(0.0, 360.0, n_bins + 1, dtype=np.float32)[1:-1].tolist()
    return _digitize_to_uint(wrapped, edges)


def _neighbor_stats(xyz: np.ndarray, cutoff: float = 8.0, local_seq_sep: int = 8):
    length = xyz.shape[0]
    if length == 0:
        z = np.zeros(0, dtype=np.int16)
        return z, z, z

    tree = cKDTree(xyz)
    neigh = tree.query_ball_point(xyz, r=cutoff)

    total = np.zeros(length, dtype=np.int16)
    local = np.zeros(length, dtype=np.int16)
    global_ = np.zeros(length, dtype=np.int16)

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


def _make_range_bins(total: np.ndarray, global_: np.ndarray) -> np.ndarray:
    ratio = global_ / np.maximum(total, 1)
    out = np.zeros_like(total, dtype=np.int16)
    out[(ratio >= 0.33) & (ratio <= 0.66)] = 1
    out[ratio > 0.66] = 2
    return out


def _make_burial_bins(total: np.ndarray) -> np.ndarray:
    out = np.zeros_like(total, dtype=np.int16)
    out[total >= 8] = 1
    out[total >= 16] = 2
    return out


def _compute_wcn(cb_xyz: np.ndarray, cutoff: float = 15.0) -> np.ndarray:
    if cb_xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    dmat = distance.cdist(cb_xyz, cb_xyz)
    mask = (dmat > 1e-8) & (dmat <= cutoff)
    out = np.zeros(dmat.shape[0], dtype=np.float32)
    if np.any(mask):
        inv = np.zeros_like(dmat, dtype=np.float32)
        inv[mask] = 1.0 / (dmat[mask] ** 2)
        out = inv.sum(axis=1)
    return out


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


def _write_heterodata_group(structs_group, hetero_data):
    identifier = hetero_data.identifier
    struct_group = structs_group.create_group(identifier)

    node_group = struct_group.create_group('node')
    for node_type in hetero_data.node_types:
        if hetero_data[node_type].x is not None:
            type_group = node_group.create_group(node_type)
            type_group.create_dataset('x', data=hetero_data[node_type].x.numpy())

    edge_group = struct_group.create_group('edge')
    for edge_type in hetero_data.edge_types:
        edge_name = f'{edge_type[0]}_{edge_type[1]}_{edge_type[2]}'
        type_group = edge_group.create_group(edge_name)
        if hetero_data[edge_type].edge_index is not None:
            type_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
        if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
            type_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())


class PDB2PyG:
    def __init__(self, aapropcsv: str = './foldtree2/config/aaindex1.csv'):
        self.aapropcsv = aapropcsv
        aaproperties = pd.read_csv(aapropcsv, header=0)
        colmap = {aaproperties.columns[i]: i for i in range(len(aaproperties.columns))}
        aaproperties.drop(['description', 'reference'], axis=1, inplace=True)
        onehot = pd.get_dummies(aaproperties.columns.unique()).astype(int)
        aaindex = {c: onehot[c].argmax() for c in onehot.columns}
        aaproperties = pd.concat([aaproperties, onehot], axis=0)
        aaproperties = aaproperties.T
        aaproperties[aaproperties.isna()] = 0

        self.aaproperties = aaproperties
        self.onehot = onehot
        self.colmap = colmap
        self.aaindex = aaindex
        self.revmap_aa = {v: k for k, v in aaindex.items()}

    @staticmethod
    def read_structure(path: str) -> gemmi.Structure:
        return gemmi.read_structure(path)

    @staticmethod
    def _pick_polymer_chain(model: gemmi.Model):
        best = None
        best_len = 0
        for chain in model:
            residues = []
            for res in chain:
                aa = AA3_TO_1.get(res.name.upper(), 'X')
                if aa == 'X':
                    continue
                ca = _atom_pos(res, 'CA')
                if ca is None:
                    continue
                residues.append(res)
            if len(residues) > best_len:
                best = (chain, residues)
                best_len = len(residues)
        return best

    @staticmethod
    def get_positional_encoding(seq_len: int, d_model: int):
        positional_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        position = np.arange(0, seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10.0) / d_model))
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        return positional_encoding.astype(np.float32)

    @staticmethod
    def get_sliding_window(seq_len: int, window: int = 2):
        adjacent = np.zeros((seq_len, seq_len), dtype=np.float32)
        for i in range(seq_len):
            lo = max(0, i - window)
            hi = min(seq_len, i + window + 1)
            adjacent[i, lo:hi] = 1.0
        return adjacent

    @staticmethod
    def get_backbone(chainlen: int):
        backbone = np.zeros((chainlen, chainlen), dtype=np.float32)
        backbone_rev = np.zeros((chainlen, chainlen), dtype=np.float32)
        if chainlen > 1:
            idx = np.arange(chainlen - 1)
            backbone[idx, idx + 1] = 1.0
            backbone_rev[idx + 1, idx] = 1.0
        return backbone, backbone_rev

    @staticmethod
    def keep_corner_triangles(matrix, size=30):
        mask = np.zeros_like(matrix, dtype=bool)
        n, m = matrix.shape
        size = min(size, n, m)

        for i in range(size):
            for j in range(size):
                if i + j <= size - 1:
                    mask[i, j] = True

        for i in range(size):
            for j in range(m - size, m):
                if i <= (j - (m - size)):
                    mask[i, j] = True

        for i in range(n - size, n):
            for j in range(size):
                if (i - (n - size)) >= j:
                    mask[i, j] = True

        for i in range(n - size, n):
            for j in range(m - size, m):
                if (i - (n - size)) + (j - (m - size)) >= size - 1:
                    mask[i, j] = True

        values = matrix[mask]
        values = np.expand_dims(values, axis=0)
        result = np.zeros_like(matrix)
        result[mask] = matrix[mask]
        return result, values

    @staticmethod
    def sparse2pairs(sparsemat):
        row, col, _ = scipy.sparse.find(sparsemat)
        return np.vstack([row, col])

    def add_aaproperties(self, angles_df: pd.DataFrame):
        nodeprops = angles_df.merge(self.aaproperties, left_on='single_letter_code', right_index=True, how='left')
        nodeprops = nodeprops.replace(np.nan, 0)
        return nodeprops

    def _extract_chain_arrays(self, chain: gemmi.Chain):
        residues = []
        residue_names = []
        aa_chars = []
        ca_list = []
        cb_list = []
        n_list = []
        c_list = []
        o_list = []
        sg_list = []
        plddt_list = []
        chi_list = []
        sc_centroid_list = []

        for res in chain:
            aa = AA3_TO_1.get(res.name.upper(), 'X')
            if aa == 'X':
                continue
            ca = _atom_pos(res, 'CA')
            if ca is None:
                continue

            n = _atom_pos(res, 'N')
            c = _atom_pos(res, 'C')
            o = _atom_pos(res, 'O')
            cb = _atom_pos(res, 'CB')
            if cb is None:
                if n is not None and c is not None:
                    cb = _pseudo_cb_from_ca_n_c(ca, n, c)
                else:
                    cb = ca

            if n is None:
                n = ca
            if c is None:
                c = ca
            if o is None:
                o = ca

            plddt = _atom_bfactor(res, 'CA')
            if plddt is None:
                bfs = [float(atom.b_iso) for atom in res]
                plddt = float(np.mean(bfs)) if bfs else 0.0
            sg = _atom_pos(res, 'SG')
            if sg is None:
                sg = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

            # Compute chi angles and sidechain centroid
            res_name = res.name.upper()
            chi_angles = _compute_chi_angles(res, res_name)
            sc_centroid = _compute_sidechain_centroid(res, res_name, ca)

            residues.append(res)
            residue_names.append(res_name)
            aa_chars.append(aa)
            ca_list.append(ca)
            cb_list.append(cb)
            n_list.append(n)
            c_list.append(c)
            o_list.append(o)
            sg_list.append(sg)
            plddt_list.append(plddt)
            chi_list.append(chi_angles)
            sc_centroid_list.append(sc_centroid)

        if not residues:
            return None

        return {
            'residues': residues,
            'residue_names': residue_names,
            'aa': ''.join(aa_chars),
            'ca': np.asarray(ca_list, dtype=np.float32),
            'cb': np.asarray(cb_list, dtype=np.float32),
            'n': np.asarray(n_list, dtype=np.float32),
            'c': np.asarray(c_list, dtype=np.float32),
            'o': np.asarray(o_list, dtype=np.float32),
            'sg': np.asarray(sg_list, dtype=np.float32),
            'plddt': np.asarray(plddt_list, dtype=np.float32),
            'chi': np.asarray(chi_list, dtype=np.float32),  # Shape: (N, MAX_CHI*2)
            'sc_centroid': np.asarray(sc_centroid_list, dtype=np.float32),  # Shape: (N, 4)
        }

    def _gemmi_angles(self, chain: gemmi.Chain, residues: list[gemmi.Residue]):
        rows = []
        for res in residues:
            prev_res = chain.previous_residue(res)
            next_res = chain.next_residue(res)

            phi = 0.0
            psi = 0.0
            omega = 0.0

            try:
                if prev_res is not None and next_res is not None:
                    p, s = gemmi.calculate_phi_psi(prev_res, res, next_res)
                    phi = float(np.degrees(p)) if np.isfinite(p) else 0.0
                    psi = float(np.degrees(s)) if np.isfinite(s) else 0.0
            except Exception:
                phi = 0.0
                psi = 0.0

            try:
                if next_res is not None:
                    w = gemmi.calculate_omega(res, next_res)
                    omega = float(np.degrees(w)) if np.isfinite(w) else 0.0
            except Exception:
                omega = 0.0

            seqid = int(res.seqid.num)
            rows.append({
                'Chain': chain.name,
                'Residue_Number': seqid,
                'Residue_Name': res.name,
                'single_letter_code': AA3_TO_1.get(res.name.upper(), 'X'),
                'Phi_Angle': phi,
                'Psi_Angle': psi,
                'Omega_Angle': omega,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _angles_to_ss_onehot(phi_psi_omega: np.ndarray) -> np.ndarray:
        length = phi_psi_omega.shape[0]
        ss = np.zeros((length, 3), dtype=np.float32)
        for i in range(length):
            phi = float(phi_psi_omega[i, 0])
            psi = float(phi_psi_omega[i, 1])

            if (-160.0 <= phi <= -20.0) and (-90.0 <= psi <= 45.0):
                ss[i, 0] = 1.0
            elif (-180.0 <= phi <= -40.0) and (psi >= 90.0 or psi <= -120.0):
                ss[i, 1] = 1.0
            else:
                ss[i, 2] = 1.0
        return ss

    def _compute_ss(self, phi_psi_omega):
        return self._angles_to_ss_onehot(phi_psi_omega)

    def _compute_contact_matrix(self, xyz: np.ndarray, distance_cutoff: float = 8.0):
        dmat = distance.cdist(xyz, xyz)
        mask = (dmat > 1e-8) & (dmat < distance_cutoff)
        out = np.zeros_like(dmat, dtype=np.float32)
        out[mask] = dmat[mask]
        return out

    def _compute_bond_type_maps(
        self,
        ca_xyz: np.ndarray,
        cb_xyz: np.ndarray,
        residue_names: list[str],
        sg_xyz: np.ndarray,
        n_xyz: np.ndarray,
        o_xyz: np.ndarray,
        c_xyz: np.ndarray,
        peptide_cutoff: float = 4.2,
        contact_cutoff: float = 8.0,
        pi_cutoff: float = 5.5,
        ionic_cutoff: float = 6.0,
        disulfide_cutoff: float = 2.5,
        hbond_cutoff: float = 3.5,
        hbond_angle_cutoff: float = 120.0,
    ):
        """
        Compute various bond type maps between residues.
        
        Returns:
            peptide_map, contact_map, pi_stacking_map, ionic_map, disulfide_map, hbond_map
        """
        length = ca_xyz.shape[0]
        peptide_map = np.zeros((length, length), dtype=np.float32)
        contact_map = np.zeros((length, length), dtype=np.float32)
        pi_stacking_map = np.zeros((length, length), dtype=np.float32)
        ionic_map = np.zeros((length, length), dtype=np.float32)
        disulfide_map = np.zeros((length, length), dtype=np.float32)
        hbond_map = np.zeros((length, length), dtype=np.float32)
        
        if length == 0:
            return peptide_map, contact_map, pi_stacking_map, ionic_map, disulfide_map, hbond_map

        ca_dmat = distance.cdist(ca_xyz, ca_xyz)
        cb_dmat = distance.cdist(cb_xyz, cb_xyz)
        valid_sg = np.all(np.isfinite(sg_xyz), axis=1)
        sg_dmat = distance.cdist(sg_xyz, sg_xyz)
        
        # N-O distance matrix for hydrogen bond detection
        # Backbone H-bond: N-H...O=C where N is donor and O is acceptor
        # N of residue i donates to O of residue j
        n_o_dmat = distance.cdist(n_xyz, o_xyz)

        aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        cationic = {'ARG', 'LYS', 'HIS'}
        anionic = {'ASP', 'GLU'}

        # Peptide bonds: only immediate sequence neighbors with short CA-CA distance.
        for i in range(length - 1):
            j = i + 1
            if ca_dmat[i, j] <= peptide_cutoff:
                peptide_map[i, j] = 1.0
                peptide_map[j, i] = 1.0

        # Contacts: close in 3D, but not immediate sequence neighbors.
        contact_mask = (cb_dmat > 1e-8) & (cb_dmat <= contact_cutoff)
        
        # Vectorized angle computation for H-bonds
        # For H-bond from N[i] to O[j], check angle at N: C[i-1]-N[i]-O[j]
        # This approximates the N-H...O angle without explicit H positions
        cos_angle_cutoff = np.cos(np.radians(180.0 - hbond_angle_cutoff))
        
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                if abs(i - j) <= 1:
                    continue
                if contact_mask[i, j]:
                    contact_map[i, j] = 1.0

                # Pi-stacking proxy: aromatic pair with short sidechain-center (CB) distance.
                if (
                    residue_names[i] in aromatic
                    and residue_names[j] in aromatic
                    and cb_dmat[i, j] <= pi_cutoff
                ):
                    pi_stacking_map[i, j] = 1.0

                # Ionic proxy: opposite-charge pair with short sidechain-center distance.
                is_ionic_pair = (
                    (residue_names[i] in cationic and residue_names[j] in anionic)
                    or (residue_names[j] in cationic and residue_names[i] in anionic)
                )
                if is_ionic_pair and cb_dmat[i, j] <= ionic_cutoff:
                    ionic_map[i, j] = 1.0

                # Disulfide: CYS-CYS with valid SG atoms and SG-SG distance threshold.
                if (
                    residue_names[i] == 'CYS'
                    and residue_names[j] == 'CYS'
                    and valid_sg[i]
                    and valid_sg[j]
                    and sg_dmat[i, j] <= disulfide_cutoff
                ):
                    disulfide_map[i, j] = 1.0
                
                # Hydrogen bonds: backbone N-H...O=C
                # N of residue i donates to O of residue j
                # Distance criterion: N-O distance < hbond_cutoff (typically 3.5Å)
                # Angle criterion: approximate N-H...O angle using C-N...O angle
                if n_o_dmat[i, j] <= hbond_cutoff:
                    # Check angle: use C[i]-N[i]-O[j] as proxy for N-H...O angle
                    # Since H is approximately along C->N direction extended
                    if i > 0:
                        # Vector from C of previous residue to N of current
                        c_to_n = n_xyz[i] - c_xyz[i - 1]
                        c_to_n_norm = np.linalg.norm(c_to_n)
                        if c_to_n_norm > 1e-8:
                            c_to_n = c_to_n / c_to_n_norm
                            
                            # Vector from N to O
                            n_to_o = o_xyz[j] - n_xyz[i]
                            n_to_o_norm = np.linalg.norm(n_to_o)
                            if n_to_o_norm > 1e-8:
                                n_to_o = n_to_o / n_to_o_norm
                                
                                # Angle between C-N and N-O vectors
                                # H is roughly opposite to C, so we want angle > 120° (cos < -0.5)
                                cos_angle = np.dot(c_to_n, n_to_o)
                                # We want the N-H...O angle, H is opposite to C direction
                                # So the actual H-N-O angle is ~180 - acos(cos_angle)
                                # For good H-bond: N-H...O > 120°, meaning H-N-O < 60°
                                # But since H is opposite C: C-N-O should be > 120°
                                if cos_angle <= cos_angle_cutoff:
                                    hbond_map[i, j] = 1.0
                    else:
                        # For first residue, just use distance criterion
                        hbond_map[i, j] = 1.0

        return peptide_map, contact_map, pi_stacking_map, ionic_map, disulfide_map, hbond_map

    def _fft_tracks(self, xyz: np.ndarray, cutoff_1d: int = 80, cutoff_2d: int = 25):
        dist_matrix = distance.cdist(xyz, xyz)

        fft_1d = np.fft.fft(dist_matrix, axis=1)
        if fft_1d.shape[1] < cutoff_1d:
            new = np.zeros((fft_1d.shape[0], cutoff_1d), dtype=np.complex64)
            new[:, :fft_1d.shape[1]] = fft_1d
            fft_1d = new
        else:
            fft_1d = fft_1d[:, :cutoff_1d]

        fft_2d = np.fft.fft2(dist_matrix)
        _, fft_2d_corners = self.keep_corner_triangles(fft_2d, size=cutoff_2d)
        if fft_2d_corners.shape[1] < 1300:
            new = np.zeros((1, 1300), dtype=np.complex64)
            new[:, :fft_2d_corners.shape[1]] = fft_2d_corners
            fft_2d_corners = new
        else:
            fft_2d_corners = fft_2d_corners[:, :1300]

        fft_m = np.abs(fft_1d).astype(np.float32)
        fft_a = np.angle(fft_1d).astype(np.float32)
        fft2_m = np.abs(fft_2d_corners).astype(np.float32)
        fft2_a = np.angle(fft_2d_corners).astype(np.float32)

        fft2_a = np.resize(fft2_a, fft2_m.shape)
        fft_a = np.resize(fft_a, fft_m.shape)
        return fft_m, fft_a, fft2_m, fft2_a

    @staticmethod
    def compute_local_frame(coords):
        """
        Compute local residue frames from backbone atom coordinates.
        
        Args:
            coords: [N, 3, 3] tensor where coords[:, 0, :] = N atoms,
                    coords[:, 1, :] = CA atoms, coords[:, 2, :] = C atoms
        
        Returns:
            r_true: [N, 3, 3] rotation matrices (local frames)
            t_true: [N, 3] translation vectors (CA to next CA)
            q_true: [N, 4] quaternions representing the same rotations
        """
        # Build orthonormal frame using Gram-Schmidt orthogonalization
        # x_axis: CA -> C direction (along peptide bond)
        x_axis = coords[:, 2, :] - coords[:, 1, :]
        x_norm = torch.norm(x_axis, dim=-1, keepdim=True)
        x_axis = x_axis / torch.clamp(x_norm, min=1e-8)

        # y_axis: start with CA -> N, then orthogonalize against x_axis
        y_axis = coords[:, 0, :] - coords[:, 1, :]
        # Gram-Schmidt: y = y - (y·x)x
        y_axis = y_axis - (torch.sum(y_axis * x_axis, dim=-1, keepdim=True) * x_axis)
        y_norm = torch.norm(y_axis, dim=-1, keepdim=True)
        y_axis = y_axis / torch.clamp(y_norm, min=1e-8)

        # z_axis: cross product to complete right-handed frame
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_norm = torch.norm(z_axis, dim=-1, keepdim=True)
        z_axis = z_axis / torch.clamp(z_norm, min=1e-8)

        # Stack axes as columns to form rotation matrix [N, 3, 3]
        # Each row of r_true[i] transforms from local to global coordinates
        r_true = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        # Translation: vector from current CA to next CA
        t_true = coords[1:, 1, :] - coords[:-1, 1, :]
        t_pad = torch.zeros((1, 3), device=coords.device, dtype=coords.dtype)
        t_true = torch.cat([t_true, t_pad], dim=0)
        
        # Convert rotation matrices to quaternions using rigid_utils
        # rot_to_quat expects [*, 3, 3] and returns [*, 4] quaternions (w, x, y, z)
        q_true = rot_to_quat(r_true)
        
        return r_true, t_true, q_true

    def create_features(self, pdb_file: str, distance_cutoff: float = 8.0):
        st = self.read_structure(pdb_file)
        if len(st) == 0:
            return None
        model = st[0]
        picked = self._pick_polymer_chain(model)
        if picked is None:
            return None
        chain, residues = picked

        arr = self._extract_chain_arrays(chain)
        if arr is None:
            return None

        angles = self._gemmi_angles(chain, residues)
        if len(angles) == 0:
            return None

        angles = self.add_aaproperties(angles)
        angles = angles.dropna().reset_index(drop=True)
        angles = angles.drop(['Chain', 'Residue_Number', 'Residue_Name'], axis=1)

        positional_encoding = self.get_positional_encoding(len(arr['ca']), 256)
        angles_with_pe = pd.concat([angles, pd.DataFrame(positional_encoding)], axis=1)

        aa = np.array(angles.iloc[:, -20:], dtype=np.float32)
        bondangles = np.array(angles[['Phi_Angle', 'Psi_Angle', 'Omega_Angle']], dtype=np.float32)

        contact_points = self._compute_contact_matrix(arr['cb'], distance_cutoff=distance_cutoff)
        ss = self._compute_ss(phi_psi_omega=bondangles)

        backbone, backbone_rev = self.get_backbone(len(arr['ca']))
        window = self.get_sliding_window(len(arr['ca']), window=2)
        window_rev = window.T

        total, local, global_ = _neighbor_stats(arr['cb'], cutoff=8.0, local_seq_sep=8)
        range_bins = _make_range_bins(total, global_).astype(np.float32)
        burial_bins = _make_burial_bins(total).astype(np.float32)
        bend_bins = _make_bend_bins(arr['ca'], n_bins=8).astype(np.float32)
        torsion_bins = _make_torsion_bins(arr['ca'], n_bins=12).astype(np.float32)
        wcn = _compute_wcn(arr['cb'], cutoff=15.0).astype(np.float32)

        track_features = {
            'contact_number': total.astype(np.float32),
            'local_contacts': local.astype(np.float32),
            'global_contacts': global_.astype(np.float32),
            'range_bin': range_bins,
            'burial_bin': burial_bins,
            'bend_bin': bend_bins,
            'torsion_bin': torsion_bins,
            'wcn': wcn,
        }

        fft1r, fft1i, fft2r, fft2i = self._fft_tracks(arr['cb'], cutoff_1d=80, cutoff_2d=25)
        peptide_bond_map, contact_bond_map, pi_stacking_map, ionic_map, disulfide_map, hbond_map = self._compute_bond_type_maps(
            ca_xyz=arr['ca'],
            cb_xyz=arr['cb'],
            residue_names=arr['residue_names'],
            sg_xyz=arr['sg'],
            n_xyz=arr['n'],
            o_xyz=arr['o'],
            c_xyz=arr['c'],
            peptide_cutoff=4.2,
            contact_cutoff=distance_cutoff,
        )

        return {
            'identifier': f"{Path(pdb_file).stem}_{chain.name}",
            'angles': angles_with_pe,
            'aa': aa,
            'bondangles': bondangles,
            'contact_points': contact_points,
            'backbone': backbone,
            'backbone_rev': backbone_rev,
            'window': window,
            'window_rev': window_rev,
            'ss': ss,
            'peptide_bond_map': peptide_bond_map,
            'contact_bond_map': contact_bond_map,
            'pi_stacking_map': pi_stacking_map,
            'ionic_map': ionic_map,
            'disulfide_map': disulfide_map,
            'hbond_map': hbond_map,
            'plddt': (arr['plddt'] / 100.0).reshape(-1, 1).astype(np.float32),
            'coords': arr['ca'].astype(np.float32),
            'ncoords': arr['n'].astype(np.float32),
            'ccoords': arr['c'].astype(np.float32),
            'ocoords': arr['o'].astype(np.float32),
            'cbcoords': arr['cb'].astype(np.float32),
            'positional_encoding': positional_encoding.astype(np.float32),
            'fft1r': fft1r,
            'fft1i': fft1i,
            'fft2r': fft2r,
            'fft2i': fft2i,
            'track_features': track_features,
            'chi': arr['chi'].astype(np.float32),  # Shape: (N, MAX_CHI*2) = (N, 8)
            'sc_centroid': arr['sc_centroid'].astype(np.float32),  # Shape: (N, 4)
        }

    def struct2pyg(self, pdbchain, identifier=None, include_chain=False, verbose=False, compute_hbonds=True, **kwargs):
        del verbose
        del kwargs
        del compute_hbonds

        if not isinstance(pdbchain, str):
            raise ValueError('pdbgraphmk2 expects pdbchain as a path string')

        feat = self.create_features(pdbchain)
        if feat is None:
            return None

        data = HeteroData()
        if identifier is None:
            identifier = feat['identifier'] if include_chain else Path(pdbchain).stem
        data.identifier = identifier

        angles = feat['angles'].drop(['single_letter_code'], axis=1)
        angles = angles.fillna(0.0)

        residue_frames = torch.stack([
            torch.tensor(feat['ncoords'], dtype=torch.float32),
            torch.tensor(feat['coords'], dtype=torch.float32),
            torch.tensor(feat['ccoords'], dtype=torch.float32),
        ], dim=1)
        r_true, t_true, q_true = self.compute_local_frame(residue_frames)

        data['AA'].x = torch.tensor(feat['aa'], dtype=torch.float32)
        data['coords'].x = torch.tensor(feat['coords'], dtype=torch.float32)
        data['cbcoords'].x = torch.tensor(feat['cbcoords'], dtype=torch.float32)
        data['R_true'].x = r_true
        data['t_true'].x = t_true
        data['q_true'].x = q_true  # Quaternion representation of rotation (w, x, y, z)
        data['bondangles'].x = torch.tensor(feat['bondangles'], dtype=torch.float32)
        data['plddt'].x = torch.tensor(feat['plddt'], dtype=torch.float32)
        data['positions'].x = torch.tensor(feat['positional_encoding'], dtype=torch.float32)
        data['ss'].x = torch.tensor(feat['ss'], dtype=torch.float32)
        data['chi'].x = torch.tensor(feat['chi'], dtype=torch.float32)  # Shape: (N, 8) - sin/cos for chi1-4
        data['sc_centroid'].x = torch.tensor(feat['sc_centroid'], dtype=torch.float32)  # Shape: (N, 4) - dir_xyz + distance

        track_names = ['contact_number', 'local_contacts', 'global_contacts', 'range_bin', 'burial_bin', 'bend_bin', 'torsion_bin', 'wcn']
        track_stack = []
        for key in track_names:
            vals = feat['track_features'][key].reshape(-1, 1)
            data[key].x = torch.tensor(vals, dtype=torch.float32)
            track_stack.append(vals)

        base_angles = torch.tensor(angles.values, dtype=torch.float32)
        track_mat = torch.tensor(np.concatenate(track_stack, axis=1), dtype=torch.float32)
        data['res'].x = torch.cat([base_angles, r_true.view(-1, 9), t_true, track_mat], dim=1)

        data['fourier1dr'].x = torch.tensor(feat['fft1r'], dtype=torch.float32)
        data['fourier1di'].x = torch.tensor(feat['fft1i'], dtype=torch.float32)
        data['fourier2dr'].x = torch.tensor(feat['fft2r'], dtype=torch.float32)
        data['fourier2di'].x = torch.tensor(feat['fft2i'], dtype=torch.float32)

        data['godnode'].x = torch.ones((1, 5), dtype=torch.float32)
        data['godnode4decoder'].x = torch.ones((1, 5), dtype=torch.float32)

        contact_points = sparse.csr_matrix(feat['contact_points'])
        backbone = sparse.csr_matrix(feat['backbone'])
        backbone_rev = sparse.csr_matrix(feat['backbone_rev'])
        window = sparse.csr_matrix(feat['window'])
        window_rev = sparse.csr_matrix(feat['window_rev'])

        # Edge attribute layout: [distance_or_weight, relation_onehot_5..., bond_type_onehot_6...]
        # relation_onehot_5 index mapping: 0=backbone, 1=backbonerev, 2=contactPoints, 3=window, 4=windowrev
        # bond_type_onehot_6 index mapping:
        # 0=peptide_bond, 1=contact, 2=pi_stacking, 3=ionic, 4=disulfide, 5=hbond
        bond_type_names = ['backbone', 'backbonerev', 'contactPoints', 'window', 'windowrev']
        bond_type_index = {name: idx for idx, name in enumerate(bond_type_names)}
        chem_bond_type_names = ['peptide_bond', 'contact', 'pi_stacking', 'ionic', 'disulfide', 'hbond']

        peptide_bond_map = feat['peptide_bond_map']
        contact_bond_map = feat['contact_bond_map']
        pi_stacking_map = feat['pi_stacking_map']
        ionic_map = feat['ionic_map']
        disulfide_map = feat['disulfide_map']
        hbond_map = feat['hbond_map']

        def build_edge_attr(values: np.ndarray, edge_name: str, edge_index: np.ndarray) -> torch.Tensor:
            scalar = torch.tensor(values, dtype=torch.float32).view(-1, 1)
            relation_onehot = torch.zeros((scalar.shape[0], len(bond_type_names)), dtype=torch.float32)
            bond_onehot = torch.zeros((scalar.shape[0], len(chem_bond_type_names)), dtype=torch.float32)

            if scalar.shape[0] > 0:
                relation_onehot[:, bond_type_index[edge_name]] = 1.0
                if edge_name in {'backbone', 'backbonerev', 'contactPoints'}:
                    src = edge_index[0]
                    dst = edge_index[1]
                    peptide_vals = peptide_bond_map[src, dst]
                    contact_vals = contact_bond_map[src, dst]
                    pi_vals = pi_stacking_map[src, dst]
                    ionic_vals = ionic_map[src, dst]
                    disulfide_vals = disulfide_map[src, dst]
                    hbond_vals = hbond_map[src, dst]
                    bond_onehot[:, 0] = torch.tensor(peptide_vals, dtype=torch.float32)
                    bond_onehot[:, 1] = torch.tensor(contact_vals, dtype=torch.float32)
                    bond_onehot[:, 2] = torch.tensor(pi_vals, dtype=torch.float32)
                    bond_onehot[:, 3] = torch.tensor(ionic_vals, dtype=torch.float32)
                    bond_onehot[:, 4] = torch.tensor(disulfide_vals, dtype=torch.float32)
                    bond_onehot[:, 5] = torch.tensor(hbond_vals, dtype=torch.float32)

            return torch.cat([scalar, relation_onehot, bond_onehot], dim=1)

        backbone_idx = self.sparse2pairs(backbone)
        backbone_rev_idx = self.sparse2pairs(backbone_rev)
        contact_idx = self.sparse2pairs(contact_points)
        window_idx = self.sparse2pairs(window)
        window_rev_idx = self.sparse2pairs(window_rev)

        data['res', 'backbone', 'res'].edge_index = torch.tensor(backbone_idx, dtype=torch.long)
        data['res', 'backbonerev', 'res'].edge_index = torch.tensor(backbone_rev_idx, dtype=torch.long)
        data['res', 'contactPoints', 'res'].edge_index = torch.tensor(contact_idx, dtype=torch.long)
        data['res', 'window', 'res'].edge_index = torch.tensor(window_idx, dtype=torch.long)
        data['res', 'windowrev', 'res'].edge_index = torch.tensor(window_rev_idx, dtype=torch.long)

        data['res', 'backbone', 'res'].edge_attr = build_edge_attr(backbone.data, 'backbone', backbone_idx)
        data['res', 'backbonerev', 'res'].edge_attr = build_edge_attr(backbone_rev.data, 'backbonerev', backbone_rev_idx)
        data['res', 'contactPoints', 'res'].edge_attr = build_edge_attr(contact_points.data, 'contactPoints', contact_idx)
        data['res', 'window', 'res'].edge_attr = build_edge_attr(window.data, 'window', window_idx)
        data['res', 'windowrev', 'res'].edge_attr = build_edge_attr(window_rev.data, 'windowrev', window_rev_idx)

        n_res = data['res'].x.shape[0]
        sparse_godnode = np.vstack([np.arange(n_res), np.zeros(n_res, dtype=np.int64)])
        sparse_godnode2res = np.vstack([np.zeros(n_res, dtype=np.int64), np.arange(n_res)])

        data['res', 'informs', 'godnode'].edge_index = torch.tensor(sparse_godnode, dtype=torch.long)
        data['godnode', 'informs', 'res'].edge_index = torch.tensor(sparse_godnode2res, dtype=torch.long)
        data['res', 'informs', 'godnode4decoder'].edge_index = torch.tensor(sparse_godnode, dtype=torch.long)
        data['godnode4decoder', 'informs', 'res'].edge_index = torch.tensor(sparse_godnode2res, dtype=torch.long)

        data['res', 'contactPoints', 'res'].edge_index, data['res', 'contactPoints', 'res'].edge_attr = torch_geometric.utils.to_undirected(
            data['res', 'contactPoints', 'res'].edge_index,
            data['res', 'contactPoints', 'res'].edge_attr,
        )
        data['res', 'backbone', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res', 'backbone', 'res'].edge_index)[0]
        data['res', 'backbonerev', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res', 'backbonerev', 'res'].edge_index)[0]

        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                data[node_type].x[torch.isnan(data[node_type].x)] = 0

        return data

    def process_single_pdb(self, pdb_file, **kwargs):
        try:
            hetero_data = self.struct2pyg(pdb_file, **kwargs)
            return hetero_data, pdb_file, None
        except Exception:
            return None, pdb_file, traceback.format_exc()

    def store_pyg(self, pdbfiles, filename, verbose=True, **kwargs):
        failed = []
        with h5py.File(filename, mode='w') as f:
            structs_group = f.create_group('structs')
            for pdb_file in tqdm.tqdm(pdbfiles):
                try:
                    hetero_data = self.struct2pyg(pdb_file, **kwargs)
                    if hetero_data is None:
                        failed.append((pdb_file, 'No data returned'))
                        continue
                    _write_heterodata_group(structs_group, hetero_data)
                except Exception as e:
                    failed.append((pdb_file, str(e)))
                    if verbose:
                        print(traceback.format_exc())
        if verbose:
            print(f'Successfully stored {len(pdbfiles) - len(failed)}/{len(pdbfiles)} structures')
        return failed

    def store_pyg_mp_pool(self, pdbfiles, filename, ncpu=4, verbose=False, chunksize=4, start_method=None, debug_log_path=None, debug_log_flush=False, **kwargs):
        args_list = [p for p in pdbfiles]
        failed_files = []
        successful_count = 0
        debug_log = open(debug_log_path, 'a') if debug_log_path else None

        def log_event(event, pdb_file, **fields):
            if debug_log is None:
                return
            payload = {'ts': time.time(), 'event': event, 'pdb_file': pdb_file}
            payload.update(fields)
            debug_log.write(json.dumps(payload) + '\n')
            if debug_log_flush:
                debug_log.flush()

        ctx = mp.get_context(start_method) if start_method else mp.get_context()

        with h5py.File(filename, mode='w') as f:
            structs_group = f.create_group('structs')
            pool = ctx.Pool(
                processes=ncpu,
                initializer=_init_pdb2pyg_worker,
                initargs=(self.aapropcsv, kwargs),
            )
            try:
                for pdb_file in args_list:
                    log_event('scheduled', pdb_file, chunksize=chunksize)

                results_iter = pool.imap_unordered(_process_pdb_worker, args_list, chunksize=chunksize)
                for hetero_data, pdb_file, error in tqdm.tqdm(results_iter, total=len(args_list), desc='Processing and storing PDB files (mk2)'):
                    if error:
                        failed_files.append((pdb_file, error))
                        log_event('process_error', pdb_file, error=error)
                        continue
                    if hetero_data is None:
                        failed_files.append((pdb_file, 'No data returned'))
                        log_event('no_data', pdb_file)
                        continue
                    try:
                        _write_heterodata_group(structs_group, hetero_data)
                        successful_count += 1
                        log_event('success', pdb_file)
                    except Exception as e:
                        failed_files.append((pdb_file, f'Storage error: {e}'))
                        log_event('store_error', pdb_file, error=str(e))
                        if verbose:
                            print(traceback.format_exc())
            finally:
                pool.close()
                pool.join()

        if debug_log is not None:
            debug_log.close()

        if verbose:
            print(f'Successfully processed and stored: {successful_count}/{len(args_list)}')
        return failed_files


_PDB2PYG_WORKER = None
_PDB2PYG_KWARGS = None


def _init_pdb2pyg_worker(aapropcsv, kwargs):
    global _PDB2PYG_WORKER, _PDB2PYG_KWARGS
    _PDB2PYG_WORKER = PDB2PyG(aapropcsv=aapropcsv)
    _PDB2PYG_KWARGS = kwargs or {}


def _process_pdb_worker(pdb_file):
    global _PDB2PYG_WORKER, _PDB2PYG_KWARGS
    try:
        hetero_data = _PDB2PYG_WORKER.struct2pyg(pdb_file, **_PDB2PYG_KWARGS)
        return hetero_data, pdb_file, None
    except Exception:
        return None, pdb_file, traceback.format_exc()


class StructureDataset(Dataset):
    def __init__(self, h5dataset):
        super().__init__()
        self.h5dataset = h5dataset
        if isinstance(h5dataset, str):
            self.h5dataset = h5py.File(h5dataset, 'r')
        self.structlist = list(self.h5dataset['structs'].keys())

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self.structlist)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            f = self.h5dataset['structs'][idx]
            identifier = idx
        else:
            identifier = self.structlist[int(idx)]
            f = self.h5dataset['structs'][identifier]

        hetero_data = HeteroData()
        hetero_data.identifier = identifier

        if 'node' in f:
            for node_type in f['node'].keys():
                node_group = f['node'][node_type]
                if 'x' in node_group:
                    hetero_data[node_type].x = torch.tensor(node_group['x'][:])

        if 'edge' in f:
            for edge_name in f['edge'].keys():
                edge_group = f['edge'][edge_name]
                src_type, link_type, dst_type = edge_name.split('_')
                edge_type = (src_type, link_type, dst_type)
                if 'edge_index' in edge_group:
                    hetero_data[edge_type].edge_index = torch.tensor(edge_group['edge_index'][:])
                if 'edge_attr' in edge_group:
                    hetero_data[edge_type].edge_attr = torch.tensor(edge_group['edge_attr'][:])

        return hetero_data
