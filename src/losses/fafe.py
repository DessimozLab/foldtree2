import torch
import einops
from typing import TypeAlias, Callable, Any

Tensor: TypeAlias = torch.Tensor

def invert_rigid(R: Tensor, t: Tensor):
    """Invert rigid transformation.

    Args:
        R: Rotation matrices, (..., 3, 3).
        t: Translation, (..., 3).

    Returns:
        R_inv: Inverted rotation matrices, (..., 3, 3).
        t_inv: Inverted translation, (..., 3).
    """
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.einsum("... r t , ... t -> ... r", R_inv, t)
    return R_inv, t_inv


def node2pair(t1: Tensor, t2: Tensor, sequence_dim: int, op: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    """
    Create a pair tensor from a single tensor

    Args:
        t1: The first tensor to be converted to pair tensor
        t2: The second tensor to be converted to pair tensor
        sequence_dim: The dimension of the sequence
        op: The operation to be applied to the pair tensor

    Returns:
        Tensor: The pair tensor

    """
    # convert to positive if necessary
    if sequence_dim < 0:
        sequence_dim = t1.ndim + sequence_dim
    if t1.ndim != t2.ndim:
        raise ValueError(f"t1 and t2 must have the same number of dimensions, got {t1.ndim} and {t2.ndim}")
    t1 = t1.unsqueeze(sequence_dim + 1)
    t2 = t2.unsqueeze(sequence_dim)
    return op(t1, t2)


def compose_rotation_and_translation(
    R1: Tensor,
    t1: Tensor,
    R2: Tensor,
    t2: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compose two frame updates.

    Ref AlphaFold2 Suppl 1.8 for details.

    Args:
        R1: Rotation of the first frames, (..., 3, 3).
        t1: Translation of the first frames, (..., 3).
        R2: Rotation of the second frames, (..., 3, 3).
        t2: Translation of the second frames, (..., 3).

    Returns:
        A tuple of new rotation and translation, (R_new, t_new).
        R_new: R1R2, (..., 3, 3).
        t_new: R1t2 + t1, (..., 3).
    """
    R_new = einops.einsum(R1, R2, "... r1 r2, ... r2 r3 -> ... r1 r3")  # (..., 3, 3)
    t_new = (
        einops.einsum(
            R1,
            t2,
            "... r t, ... t->... r",
        )
        + t1
    )  # (..., 3)

    return R_new, t_new


def masked_quadratic_mean(
    value: Tensor,
    mask: Tensor,
    dim: int | tuple[int, ...] | list[int] = -1,
    eps: float = 1e-10,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute quadratic mean value for tensor with mask.

    Args:
        value: Tensor to compute quadratic mean.
        mask: Mask of value, the same shape as `value`.
        dim: Dimension along which to compute quadratic mean.
        eps: Small number for numerical safety.
        return_masked: Whether to return masked value.

    Returns:
        Masked quadratic mean of `value`.
        [Optional] Masked value, the same shape as `value`.
    """
    return torch.sqrt((value * mask).sum(dim) / (mask.sum(dim) + eps))
    

def frame_aligned_frame_error_loss(
    R_pred: Tensor,
    t_pred: Tensor,
    R_gt: Tensor,
    t_gt: Tensor,
    frame_mask: Tensor,
    rotate_scale: float = 1.0,
    axis_scale: float = 20.0,
    eps_so3: float = 1e-7,
    eps_r3: float = 1e-4,
    dist_clamp: float | None = None,
    pair_mask: Tensor | None = None,
):
    """Compute frame aligned frame error loss with double geodesic metric.

    Args:
        R_pred: Predicted rotation matrices of frames, (..., N, 3, 3).
        t_pred: Predicted translations of frames, (..., N, 3).
        R_gt: Ground truth rotation matrices of frames, (..., N, 3, 3).
        t_gt: Ground truth translations of frames, (..., N, 3).
        frame_mask: Existing masks of ground truth frames, (..., N).
        axis_scale: Scale by which the R^3 part of loss is divided.
        eps_so3: Small number for numeric safety for arccos.
        eps_r3: Small number for numeric safety for sqrt.
        dist_clamp: Cutoff above which distance errors are disregarded.
        pair_mask: Additional pair masks of pairs which should be calculated, (..., N, M) or None.
            pair_mask=True, the FAPE loss is calculated; vice not calculated.
            If None, all pairs are calculated.

    Returns:
        Dict of (B) FAFE losses. Contains "fafe", "fafe_so3", "fafe_r3".
    """
    N = R_pred.shape[-3]

    def _diff_frame(R: Tensor, t: Tensor) -> Tensor:
        R_inv, t_inv = invert_rigid(
            R=einops.repeat(R, "... i r1 r2 -> ... (i j) r1 r2", j=N),
            t=einops.repeat(t, "... i t -> ... (i j) t", j=N),
        )
        R_j = einops.repeat(R, "... j r1 r2 -> ... (i j) r1 r2", i=N)
        t_j = einops.repeat(t, "... j t -> ... (i j) t", i=N)

        return compose_rotation_and_translation(R_inv, t_inv, R_j, t_j)

    frame_mask = node2pair(frame_mask, frame_mask, -1, torch.logical_and)
    if pair_mask is not None:
        frame_mask = pair_mask * frame_mask
    frame_mask = einops.rearrange(frame_mask, "... i j -> ... (i j)")

    losses = compute_double_geodesic_error(
        *_diff_frame(R_pred, t_pred),
        *_diff_frame(R_gt, t_gt),
        frame_mask=frame_mask,
        rotate_scale=rotate_scale,
        axis_scale=axis_scale,
        dist_clamp=dist_clamp,
        eps_so3=eps_so3,
        eps_r3=eps_r3,
    )
    return losses


def compute_double_geodesic_error(
    R_pred: Tensor,
    t_pred: Tensor,
    R_gt: Tensor,
    t_gt: Tensor,
    frame_mask: Tensor,
    rotate_scale: float = 1.0,
    axis_scale: float = 20.0,
    dist_clamp: float | None = None,
    eps_so3: float = 1e-7,
    eps_r3: float = 1e-4,
):
    """Compute frame-wise error with double geodesic metric.

    d_se3(T_pred, T_gt) = sqrt(d_so3(R_pred, R_gt)^2 + (d_r3(t_pred, t_gt) / axis_scale)^2)
    d_so3(R_pred, R_gt) range [0, pi]
    d_r3(t_pred, t_gt) / axis_scale) range [0, 1.5] when clamping
    
    Args:
        R_pred: Predicted rotation matrices of T, (..., N, 3, 3).
        t_pred: Predicted translations of T, (..., N, 3).
        R_gt: Ground truth rotation matrices of T, (..., N, 3, 3).
        t_gt: Ground truth translations of T, (..., N, 3).
        frame_mask: Existing masks of ground truth T, (..., N).
        rotate_scale: Scale by which the SO3 part of loss is divided.
        axis_scale: Scale by which the R^3 part of loss is divided.
        dist_clamp: Cutoff above which distance errors are disregarded.
        eps_so3: Small number for numeric safety for arccos.
            Refer to https://github.com/pytorch/pytorch/issues/8069
        ep3_r3: Small number for numeric safety for sqrt.

    Returns:
        Dict of (B) FAFE losses. Contains "fafe", "fafe_so3", "fafe_r3".

    Note:
        so3 loss/error presented in scaled form [0, pi/rotate_scale].
        r3 loss/error presented in scaled form [0, dist_clamp/axis_scale].
    """
    if dist_clamp is None:
        dist_clamp = 1e9

    # SO3 loss
    R_diff = einops.rearrange(R_pred, "... i j -> ... j i") @ R_gt
    R_diff_trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    so3_dist = torch.acos(torch.clamp((R_diff_trace - 1) / 2, -1 + eps_so3, 1 - eps_so3)) / rotate_scale  # (..., N)
    so3_loss = masked_quadratic_mean(so3_dist, frame_mask, dim=(-1))

    # R3 loss
    r3_dist = torch.sqrt(torch.sum((t_pred - t_gt) ** 2, dim=-1) + eps_r3)  # (..., N)
    r3_dist = r3_dist.clamp(max=dist_clamp) / axis_scale  # (..., N)
    r3_loss = masked_quadratic_mean(r3_dist, frame_mask, dim=(-1))

    # double geodesic loss
    se3_dist = torch.sqrt(so3_dist**2 + r3_dist**2)  # (..., N)
    se3_loss = masked_quadratic_mean(se3_dist, frame_mask, dim=(-1))

    losses = {
        "fafe": se3_loss,  # Note se3_loss = sqrt((so3_loss/rotate_scale)^2 + (r3_loss/axis_scale)^2)
        "fafe_so3": so3_loss,
        "fafe_r3": r3_loss,
    }

    return losses


