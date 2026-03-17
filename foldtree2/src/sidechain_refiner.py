import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Maximum number of chi torsions per amino acid.
# ALA/GLY have 0; other values follow standard sidechain definitions.
MAX_CHI_BY_AA = {
    "A": 0,
    "R": 4,
    "N": 2,
    "D": 2,
    "C": 1,
    "Q": 3,
    "E": 3,
    "G": 0,
    "H": 2,
    "I": 2,
    "L": 2,
    "K": 4,
    "M": 3,
    "F": 2,
    "P": 2,
    "S": 1,
    "T": 1,
    "W": 2,
    "Y": 2,
    "V": 1,
}

MAX_CHI = 4
AA_CHI_COUNTS = torch.tensor([MAX_CHI_BY_AA[a] for a in AA_ORDER], dtype=torch.long)
AA_HAS_CB = torch.tensor([1 if a != "G" else 0 for a in AA_ORDER], dtype=torch.bool)


class SidechainRefiner(nn.Module):
    """
    AA-conditioned sidechain geometry module for separate-stage training.

    Intended usage:
    1) Train coarse backbone stack (encoder + quaternion decoder/refiner).
    2) Freeze or partially freeze coarse stack.
    3) Train this module on top of residue latents + coarse geometry.

    Inputs
    ------
    - `res_latent`: (N, C) residue embeddings from trained encoder/refiner.
    - `aa_idx`: (N,) residue identities in [0, 19].
    - `quat_pred`: (N, 4) optional coarse quaternion from backbone stage.
    - `trans_pred`: (N, 3) optional coarse translation from backbone stage.
    - `batch`: (N,) optional batch graph ids.

    Outputs
    -------
    - `cb_dir`: (N, 3) predicted Cβ direction unit vectors in local frame.
    - `chi_sin_cos`: (N, 4, 2) predicted sin/cos for up to 4 chi angles.
    - `chi_mask`: (N, 4) valid chi-angle mask from amino-acid type.
    - `cb_mask`: (N,) valid Cβ mask (False for Gly).
    - `rotamer_logits`: (N, 4, R) optional logits if rotamer head enabled.
    """

    def __init__(
        self,
        latent_dim,
        aa_embed_dim=64,
        model_dim=512,
        nheads=8,
        nlayers=6,
        ff_mult=4,
        dropout=0.1,
        use_geometry_state=True,
        rotamer_bins=36,
        predict_rotamers=True,
    ):
        super().__init__()

        L.seed_everything(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.latent_dim = latent_dim
        self.aa_embed_dim = aa_embed_dim
        self.model_dim = model_dim
        self.use_geometry_state = use_geometry_state
        self.predict_rotamers = predict_rotamers
        self.rotamer_bins = rotamer_bins

        self.aa_embedding = nn.Embedding(len(AA_ORDER), aa_embed_dim)

        state_dim = 7 if use_geometry_state else 0  # quat(4) + trans(3)
        input_dim = latent_dim + aa_embed_dim + state_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nheads,
            dim_feedforward=model_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.cb_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 3),
        )

        self.chi_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, MAX_CHI * 2),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, MAX_CHI),
            nn.Sigmoid(),
        )

        if predict_rotamers:
            self.rotamer_head = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, MAX_CHI * rotamer_bins),
            )
        else:
            self.rotamer_head = None

    @staticmethod
    def _normalize(v, eps=1e-8):
        return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _pack_by_batch(x, batch):
        if batch is None:
            mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=x.device)
            return x.unsqueeze(0), mask

        num_graphs = int(batch.max().item()) + 1
        x_split = [x[batch == i] for i in range(num_graphs)]
        max_len = max(xi.shape[0] for xi in x_split)

        padded, mask = [], []
        for xi in x_split:
            orig_len = xi.shape[0]
            pad_len = max_len - orig_len
            if pad_len > 0:
                pad = torch.zeros(pad_len, xi.shape[1], device=xi.device, dtype=xi.dtype)
                xi = torch.cat([xi, pad], dim=0)
            padded.append(xi)
            m = torch.zeros(max_len, dtype=torch.bool, device=xi.device)
            m[:orig_len] = True
            mask.append(m)

        return torch.stack(padded, dim=0), torch.stack(mask, dim=0)

    @staticmethod
    def _unpack_by_batch(x_padded, batch, mask):
        if batch is None:
            return x_padded.squeeze(0)

        chunks = []
        for i in range(x_padded.shape[0]):
            chunks.append(x_padded[i][mask[i]])
        return torch.cat(chunks, dim=0)

    def _build_masks(self, aa_idx):
        chi_counts = AA_CHI_COUNTS.to(device=aa_idx.device)[aa_idx]
        chi_mask = (
            torch.arange(MAX_CHI, device=aa_idx.device).unsqueeze(0)
            < chi_counts.unsqueeze(1)
        )
        cb_mask = AA_HAS_CB.to(device=aa_idx.device)[aa_idx]
        return chi_mask, cb_mask

    def forward(
        self,
        res_latent,
        aa_idx,
        quat_pred=None,
        trans_pred=None,
        batch=None,
    ):
        if aa_idx.dtype != torch.long:
            aa_idx = aa_idx.long()

        aa_emb = self.aa_embedding(aa_idx)

        feats = [res_latent, aa_emb]
        if self.use_geometry_state:
            if quat_pred is None:
                quat_pred = torch.zeros(res_latent.shape[0], 4, device=res_latent.device, dtype=res_latent.dtype)
                quat_pred[:, 0] = 1.0
            if trans_pred is None:
                trans_pred = torch.zeros(res_latent.shape[0], 3, device=res_latent.device, dtype=res_latent.dtype)
            quat_pred = self._normalize(quat_pred)
            feats.append(quat_pred)
            feats.append(trans_pred)

        x = torch.cat(feats, dim=-1)
        x = self.input_proj(x)

        x_padded, valid_mask = self._pack_by_batch(x, batch)
        encoded = self.encoder(x_padded, src_key_padding_mask=~valid_mask)
        h = self._unpack_by_batch(encoded, batch, valid_mask)

        cb_dir = self._normalize(self.cb_head(h))

        chi_raw = self.chi_head(h).view(-1, MAX_CHI, 2)
        chi_sin_cos = self._normalize(chi_raw)

        chi_confidence = self.confidence_head(h)
        chi_mask, cb_mask = self._build_masks(aa_idx)

        rotamer_logits = None
        if self.rotamer_head is not None:
            rotamer_logits = self.rotamer_head(h).view(-1, MAX_CHI, self.rotamer_bins)

        return {
            "cb_dir": cb_dir,
            "chi_sin_cos": chi_sin_cos,
            "chi_confidence": chi_confidence,
            "chi_mask": chi_mask,
            "cb_mask": cb_mask,
            "rotamer_logits": rotamer_logits,
            "latent": h,
        }


def chi_sincos_loss(pred_chi_sincos, true_chi_sincos, chi_mask, eps=1e-8):
    """Masked L2 loss for periodic chi representation in sin/cos space."""
    pred = pred_chi_sincos / (pred_chi_sincos.norm(dim=-1, keepdim=True) + eps)
    true = true_chi_sincos / (true_chi_sincos.norm(dim=-1, keepdim=True) + eps)

    err = torch.sum((pred - true) ** 2, dim=-1)
    mask = chi_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (err * mask).sum() / denom


def cb_direction_loss(pred_cb_dir, true_cb_dir, cb_mask, eps=1e-8):
    """Masked cosine-direction loss for Cβ orientation (ignore Gly by mask)."""
    pred = pred_cb_dir / (pred_cb_dir.norm(dim=-1, keepdim=True) + eps)
    true = true_cb_dir / (true_cb_dir.norm(dim=-1, keepdim=True) + eps)

    cos_sim = torch.sum(pred * true, dim=-1).clamp(-1.0, 1.0)
    loss = 1.0 - cos_sim
    mask = cb_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def sidechain_total_loss(
    pred,
    target,
    chi_weight=1.0,
    cb_weight=1.0,
    conf_weight=0.1,
):
    """
    Starter multi-term sidechain loss.

    Expected dict keys:
    - pred: chi_sin_cos, cb_dir, chi_confidence, chi_mask, cb_mask
    - target: chi_sin_cos, cb_dir
    """
    chi = chi_sincos_loss(
        pred_chi_sincos=pred["chi_sin_cos"],
        true_chi_sincos=target["chi_sin_cos"],
        chi_mask=pred["chi_mask"],
    )

    cb = cb_direction_loss(
        pred_cb_dir=pred["cb_dir"],
        true_cb_dir=target["cb_dir"],
        cb_mask=pred["cb_mask"],
    )

    conf_target = pred["chi_mask"].float()
    conf = F.binary_cross_entropy(pred["chi_confidence"], conf_target)

    total = chi_weight * chi + cb_weight * cb + conf_weight * conf
    return {
        "sidechain_total": total,
        "chi_loss": chi,
        "cb_loss": cb,
        "chi_conf_loss": conf,
    }
