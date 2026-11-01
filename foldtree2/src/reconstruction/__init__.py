"""MSA-based coordinate reconstruction from ft2 sequences using axial attention.

Adapts the MSA Transformer (github.com/rmrao/msa-transformer) to reconstruct
3D geometry for sets of related ft2 structural sequences, where some have
known 3D coordinates (extant sequences) and some don't (ancestral).

Submodules
----------
axial_attention :
    Row/column self-attention and axial transformer blocks ported from the
    MSA Transformer. These process the (n_seq, n_res) 2D token layout.
msa_transformer_adapter :
    MsaReconstructionHead — the main model. Takes a batch of ft2 MSAs
    (B x N_seq x L), runs axial attention, and outputs per-sequence
    per-residue features ready for downstream coordinate prediction.
reconstruction_heads :
    Output heads that convert the axial features into distance distributions,
    angle predictions, coordinate predictions, and frame parameters.
    Reuses conventions from foldtree2.src.distance.distance_restraints
    and the SE(3) training losses where appropriate.
tree_msa_dataset :
    Builds training MSAs by sampling clades from OMA phylogenetic trees.
    Returns tokenised alignments with known/unknown coordinate masks.
"""

from .msa_transformer_adapter import MsaReconstructionHead
from .msa_se3_pipeline import MsaSe3Pipeline
from .axial_attention import RowSelfAttention, ColumnSelfAttention, AxialTransformerLayer
from .reconstruction_heads import (
    MsaDistanceHead,
    MsaContactHead,
    MsaAngleHead,
    MsaCoordinateHead,
    msa_distance_loss,
    msa_angle_loss,
    msa_coordinate_loss,
)
from .tree_msa_dataset import TreeMsaDataset, CladeSampledMsaDataset, msa_collate

__all__ = [
    "MsaReconstructionHead",
    "RowSelfAttention",
    "ColumnSelfAttention",
    "AxialTransformerLayer",
    "MsaDistanceHead",
    "MsaContactHead",
    "MsaAngleHead",
    "MsaCoordinateHead",
    "msa_distance_loss",
    "msa_angle_loss",
    "msa_coordinate_loss",
    "TreeMsaDataset",
    "CladeSampledMsaDataset",
    "msa_collate",
]