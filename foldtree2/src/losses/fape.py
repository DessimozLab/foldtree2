import torch
import torch.nn.functional as F
import einops
from typing import Callable, Any, Union
from foldtree2.src.losses.fape import *



def invert_rigid(R: torch.Tensor, t: torch.Tensor):
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


def node2pair(t1: torch.Tensor, t2: torch.Tensor, sequence_dim: int, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
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
    R1: torch.Tensor,
    t1: torch.Tensor,
    R2: torch.Tensor,
    t2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    value: torch.Tensor,
    mask: torch.Tensor,
    dim: Union[int, tuple[int, ...], list[int]] = -1,
    eps: float = 1e-10,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    frame_mask: torch.Tensor,
    rotate_scale: float = 1.0,
    axis_scale: float = 20.0,
    eps_so3: float = 1e-7,
    eps_r3: float = 1e-4,
    dist_clamp: Union[float, None] = None,
    pair_mask: Union[torch.Tensor, None] = None,
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

    def _diff_frame(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    frame_mask: torch.Tensor,
    rotate_scale: float = 1.0,
    axis_scale: float = 20.0,
    dist_clamp: Union[float, None] = None,
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


# ============================================================================
# Quaternion and Rotation/Translation Functions
# ============================================================================

def quaternion_multiply(q1, q2):
	"""
	Multiplies two quaternions in [w, x, y, z] format. Handles both individual quaternions
	and batches of quaternions.
	
	Args:
		q1 (Tensor): First quaternion(s) of shape (..., 4)
		q2 (Tensor): Second quaternion(s) of shape (..., 4)
	
	Returns:
		Tensor: Product quaternion(s) of shape (..., 4)
	"""
	# Handle both individual quaternions and batches
	w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
	w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
	
	# Hamilton product formula
	w = w1*w2 - x1*x2 - y1*y2 - z1*z2
	x = w1*x2 + x1*w2 + y1*z2 - z1*y2
	y = w1*y2 - x1*z2 + y1*w2 + z1*x2
	z = w1*z2 + x1*y2 - y1*x2 + z1*w2
	
	return torch.stack([w, x, y, z], dim=-1)


def quaternion_rotate(q, v):
	"""
	Rotates vectors using quaternions through efficient vectorized operations.
	
	Args:
		q (Tensor): Quaternions with shape (..., 4) in [w, x, y, z] format
		v (Tensor): Vectors with shape (..., 3) to rotate
		
	Returns:
		Tensor: Rotated vectors with same shape as input vectors
	"""
	# Ensure q and v have compatible batch dimensions
	if q.shape[:-1] != v.shape[:-1]:
		q = q.expand(*v.shape[:-1], 4)

	# Extract quaternion components
	w = q[..., 0]
	x = q[..., 1] 
	y = q[..., 2]
	z = q[..., 3]

	# Pre-compute common terms to avoid duplicate calculations
	ww = w * w
	xx = x * x
	yy = y * y
	zz = z * z
	wx = w * x
	wy = w * y
	wz = w * z
	xy = x * y
	xz = x * z
	yz = y * z

	# Build rotation matrix elements
	R = torch.stack([
		ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
		2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
		2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz
	], dim=-1).reshape(*q.shape[:-1], 3, 3)

	# Apply rotation
	return torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)


def quaternion_to_rotation_matrix(quat):
	"""
	Convert a batch of quaternions (w, x, y, z) into 3x3 rotation matrices.
	
	Parameters:
	- quat: (batch, N, 4) Tensor of quaternions (w, x, y, z) - scalar first

	Returns:
	- rot_matrices: (batch, N, 3, 3) Tensor of rotation matrices
	"""
	assert quat.shape[-1] == 4, "Quaternions should have shape (*, 4)"
	
	norm = torch.norm(quat, dim=-1, keepdim=True)
	quat = quat / norm
	w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]


	# Compute rotation matrix elements
	xx, yy, zz = x * x, y * y, z * z
	xy, xz, yz = x * y, x * z, y * z
	wx, wy, wz = w * x, w * y, w * z

	rot_matrices = torch.stack([
		torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
		torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
		torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
	], dim=-2)

	return rot_matrices  # Shape: (batch, N, 3, 3)


def compute_chain_positions(quaternions, translations, reference_coords=None):
	"""
	Apply rotation (quaternion) and translation to a set of 3D reference coordinates using PyTorch.
	
	Parameters:
	- quaternions: (N, 4) tensor of quaternions (w, x, y, z) - scalar first
	- translations: (N, 3) tensor of translations (tx, ty, tz)
	- reference_coords: (M, 3) tensor of reference points (default is [[0, 0, 0]])
	
	Returns:
	- transformed_coords: (N, 3) tensor of transformed coordinates
	"""
	device = quaternions.device
	quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)  # Normalize quaternions
	
	if reference_coords is None:
		reference_coords = torch.zeros(1, 3, device=device)
	
	N = quaternions.shape[0]
	
	w, x, y, z = quaternions.unbind(-1)
	
	# Rotation matrix components
	R = torch.stack([
		1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w),
		2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w),
		2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)
	], dim=-1).reshape(N, 3, 3)
	
	# Apply rotation to reference coordinates (take first point if multiple)
	rotated = torch.matmul(reference_coords[0:1], R.transpose(1,2)).squeeze(0)  # (N, 3)
	
	# Apply translation
	transformed_coords = rotated + translations
	
	return transformed_coords


def compute_chain_positions_rotmat(rotations, translations):
	"""
	Computes the global coordinates for a chain of transformations given by rotation matrices and translations.
	
	Args:
		rotations (Tensor): Shape (*, N, 3, 3) rotation matrices
		translations (Tensor): Shape (*, N, 3) translation vectors
		
	Returns:
		Tensor: Shape (*, N, 3) global coordinates for each position
	"""
	# Handle batched or unbatched input
	orig_shape = rotations.shape[:-2]
	if rotations.ndim == 3:
		rotations = rotations.unsqueeze(0)
		translations = translations.unsqueeze(0)

	batch_size = rotations.shape[0]
	N = rotations.shape[1]
	positions = []

	for b in range(batch_size):
		# Initialize starting position and rotation
		global_R = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
		curr_pos = torch.zeros(3, dtype=translations.dtype, device=translations.device)
		chain_positions = []

		for i in range(N):
			chain_positions.append(curr_pos.clone())
			# Update global rotation and position
			global_R = torch.matmul(global_R, rotations[b, i])
			curr_pos = curr_pos + torch.matmul(global_R, translations[b, i])

		positions.append(torch.stack(chain_positions))

	positions = torch.stack(positions)
	
	# Return to original shape if unbatched input
	if len(orig_shape) == 1:
		positions = positions.squeeze(0)
		
	return positions


def transform_rt_to_coordinates(rotations, translations):
	"""
	Convert R, t matrices into global 3D coordinates.
	"""
	batch_size, num_residues, _ = rotations.shape
	coords = torch.zeros((batch_size, num_residues, 3), device=rotations.device)
	for b in range(batch_size):
		transform = torch.eye(4, device=rotations.device)
		for i in range(num_residues):
			pass  # Implementation needed
	return coords


# ============================================================================
# Loss Functions for Quaternions and Rotations
# ============================================================================

def quaternion_fape_loss(q_pred, t_pred, q_target, t_target, points):
	"""
	Computes a Frame Aligned Point Error (FAPE) loss between predicted and target frames.
	Each frame is represented by a unit quaternion and a translation vector.
	
	Args:
		q_pred: Tensor of shape (batch, 4) with predicted unit quaternions.
		t_pred: Tensor of shape (batch, 3) with predicted translations.
		q_target: Tensor of shape (batch, 4) with target unit quaternions.
		t_target: Tensor of shape (batch, 3) with target translations.
		points: Tensor of shape (batch, num_points, 3) representing 3D points in a local frame.
		
	Returns:
		A scalar tensor representing the FAPE loss.
	"""
	# Normalize quaternions in case they are not perfectly unit length
	q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True)
	q_target = q_target / q_target.norm(dim=-1, keepdim=True)
	
	# Expand quaternions to apply to each point
	q_pred = q_pred.unsqueeze(1)    # shape becomes (batch, 1, 4)
	q_target = q_target.unsqueeze(1)  # shape becomes (batch, 1, 4)
	
	# Rotate the points using both the predicted and target quaternions
	points_rot_pred = quaternion_rotate(q_pred, points)   # shape: (batch, num_points, 3)
	points_rot_target = quaternion_rotate(q_target, points) # shape: (batch, num_points, 3)
	
	# Incorporate translations
	points_pred = points_rot_pred + t_pred.unsqueeze(1)     # shape: (batch, num_points, 3)
	points_target = points_rot_target + t_target.unsqueeze(1) # shape: (batch, num_points, 3)
	
	# Compute the point-wise L2 distances and then the mean error
	# Use eps inside sqrt to avoid NaN gradients when points coincide
	loss = torch.mean(torch.sqrt(torch.sum((points_pred - points_target) ** 2, dim=-1) + 1e-8))
	return loss


def fape_loss(true_R, true_t, pred_R, pred_t, batch, plddt=None, d_clamp=10.0, eps=1e-8, temperature=.25, reduction='mean', soft=False):
	"""
	Computes the Frame Aligned Point Error (FAPE) loss.
	
	For each structure in the batch, for every pair of residues (i, j),
	the local coordinates of the difference (t[j] - t[i]) are computed
	in the corresponding residue i frame using both true and predicted rotations.
	The loss is then the average clamped L2 distance between
	these local coordinates.
	
	Args:
		true_R (Tensor): True rotation matrices, shape (N, 3, 3)
		true_t (Tensor): True translation vectors, shape (N, 3)
		pred_R (Tensor): Predicted rotation matrices, shape (N, 3, 3)
		pred_t (Tensor): Predicted translation vectors, shape (N, 3)
		batch (Tensor): Batch indices for each residue, shape (N,)
		plddt (Tensor, optional): pLDDT confidence scores
		d_clamp (float, optional): Clamping threshold for error. (default: 10.0)
		eps (float, optional): Small constant for numerical stability. (default: 1e-8)
		temperature (float, optional): Temperature for soft alignment (default: 0.25)
		reduction (str, optional): Reduction method (default: 'mean')
		soft (bool, optional): Use soft alignment (default: False)
		
	Returns:
		Tensor: The scalar FAPE loss.
	""" 
	losses = []
	unique_batches = torch.unique(batch)
	for b in unique_batches:
		idx = (batch == b).nonzero(as_tuple=True)[0]
		if idx.numel() < 2:
			continue
		# Compute pairwise differences for the predicted translations.
		diff_pred = pred_t[idx].unsqueeze(1) - pred_t[idx].unsqueeze(0)  # shape: (m, m, 3)
		# Transform differences into the local predicted frames.
		local_pred = torch.einsum("mij,mnj->mni", pred_R[idx].transpose(1,2), diff_pred)
		
		# Compute pairwise differences for the true translations.
		diff_true = true_t[idx].unsqueeze(1) - true_t[idx].unsqueeze(0)  # shape: (m, m, 3)
		# Transform differences into the local true frames.
		local_true = torch.einsum("mij,mnj->mni", true_R[idx].transpose(1,2), diff_true)
		
		# Compute the L2 error per residue pair and clamp it.
		# eps is placed inside sqrt (not added to the vector) to avoid NaN gradients at zero.
		if soft == False:
			error = torch.sqrt(torch.sum((local_pred - local_true) ** 2, dim=-1) + eps)
			if plddt is not None:
				pass  # Apply pLDDT masking if needed
			error = torch.clamp(error, max=d_clamp)
			losses.append(error.mean())
		else:				
			# Compute pairwise squared Euclidean distances (manual for safe gradients)
			dist_sq = ((local_pred.unsqueeze(1) - local_true.unsqueeze(0)) ** 2).sum(dim=-1)
			dist_sq = torch.clamp(dist_sq, max=d_clamp**2)
			# Compute soft alignment probabilities using a Gaussian kernel
			soft_alignment = F.softmax(-dist_sq / temperature, dim=-1)
			# Compute soft FAPE loss
			weighted_distances = (soft_alignment * dist_sq).sum(dim=-1)  # (B, N)
			losses.append(weighted_distances.mean())

	if losses:
		return torch.stack(losses).mean()
	else:
		return torch.tensor(0.0, device=true_R.device)


def compute_lddt_loss(true_coords, pred_coords, cutoff=15.0):
	"""
	Compute lDDT loss for backpropagation.
	"""
	num_pairs = 0
	num_matching_pairs = 0
	# Manual pairwise distances with eps inside sqrt for safe gradients
	true_diff = true_coords.unsqueeze(0) - true_coords.unsqueeze(1)
	true_dists = torch.sqrt((true_diff ** 2).sum(dim=-1) + 1e-8)
	pred_diff = pred_coords.unsqueeze(0) - pred_coords.unsqueeze(1)
	pred_diff = pred_diff.clamp(-1000, 1000)  # prevent overflow when squaring
	pred_dists = torch.sqrt((pred_diff ** 2).sum(dim=-1) + 1e-8)
	mask = (true_dists < cutoff).float()
	diff = torch.abs(true_dists - pred_dists)
	valid_pairs = (diff < 0.5 * true_dists) * mask
	num_pairs += torch.sum(mask)
	num_matching_pairs += torch.sum(valid_pairs)
	lddt_score = num_matching_pairs / num_pairs if num_pairs > 0 else 0
	return 1.0 - lddt_score  # Loss formulation


def lddt_loss(coord_true, pred_R, pred_t, batch, plddt=None, d_clamp=10.0, eps=1e-8, reduction='mean'):
	"""
	Compute lDDT loss for rotation/translation predictions.
	pred_t contains CA-to-CA displacement vectors so coordinates are derived via
	reconstruct_positions (cumulative sum).
	"""
	losses = []
	unique_batches = torch.unique(batch)
	for b in unique_batches:
		idx = (batch == b).nonzero(as_tuple=True)[0]
		R_b = pred_R[idx]
		t_b = pred_t[idx]
		# coords: (N+1, 3) -> drop origin -> (N, 3)
		coord_pred = reconstruct_positions(R_b, t_b)[1:]
		c_true_b = coord_true[idx] if coord_true.shape[0] == batch.shape[0] else coord_true
		if plddt is not None:
			pmask = (plddt[idx] >= 0.3).squeeze()
			if pmask.sum() < 2:
				continue
			c_true_b = c_true_b[pmask]
			coord_pred = coord_pred[pmask]
		lddt_loss_val = compute_lddt_loss(c_true_b, coord_pred)
		losses.append(lddt_loss_val)
	if losses:
		return torch.stack(losses).mean()
	else:
		return torch.tensor(0.0, device=coord_true.device)


def differentiable_lddt_loss(
	pred_q: torch.Tensor,
	pred_t: torch.Tensor,
	true_coords: torch.Tensor,
	batch: torch.Tensor = None,
	cutoff: float = 15.0,
	thresholds: list = None,
	plddt: torch.Tensor = None,
	plddt_thresh: float = 0.3,
) -> torch.Tensor:
	"""
	Differentiable lDDT loss from predicted quaternions + CA-to-CA translations.

	Converts predicted translations to global CA positions via cumulative sum
	(since t_true[i] = CA[i+1] - CA[i] in the global frame), then evaluates
	a soft lDDT score against the ground-truth CA coordinates.

	The soft lDDT uses a Cauchy-like kernel  1 / (1 + (Δd/τ)²)  for each
	threshold τ, which is fully differentiable, smoothly rewards small errors,
	and vanishes for large errors.

	Args:
		pred_q: Predicted unit quaternions (N, 4) — unused for coordinate
		        derivation but kept for API consistency.
		pred_t: Predicted CA-to-CA displacements (N, 3).
		true_coords: Ground-truth CA coordinates (N, 3).
		batch: Per-residue batch indices (N,). None = single structure.
		cutoff: Neighbour inclusion distance in Å. Default: 15.0.
		thresholds: lDDT distance thresholds in Å.
		            Default: [0.5, 1.0, 2.0, 4.0].
		plddt: Per-residue pLDDT scores (N,) or (N, 1). Optional.
		plddt_thresh: Minimum pLDDT to include a residue. Default: 0.3.

	Returns:
		Scalar loss = 1 - mean_soft_lDDT  (lower is better).
	"""
	if thresholds is None:
		thresholds = [0.5, 1.0, 2.0, 4.0]

	def _lddt_single(t_b, coords_true_b):
		"""Compute soft lDDT loss for one structure."""
		# Derive predicted CA positions from cumulative displacements.
		# reconstruct_positions returns (N+1, 3); drop the origin row.
		pred_coords = torch.cumsum(t_b, dim=0)  # (N, 3) – directly the CA positions relative to origin
		pred_coords = pred_coords.clamp(-1000, 1000)  # prevent overflow in pairwise distances

		# Pairwise true and predicted distances, shape (N, N)
		# Manual computation with eps inside sqrt to avoid NaN gradients at zero distance.
		true_diff = coords_true_b.unsqueeze(0) - coords_true_b.unsqueeze(1)
		d_true = torch.sqrt((true_diff ** 2).sum(dim=-1) + 1e-6)
		pred_diff = pred_coords.unsqueeze(0) - pred_coords.unsqueeze(1)
		pred_diff = pred_diff.clamp(-500, 500)  # prevent overflow when squaring
		d_pred = torch.sqrt((pred_diff ** 2).sum(dim=-1) + 1e-6)
		d_pred = d_pred.clamp(1e-6, 1e6)  # prevent NaN gradients and overflow
		
		# Neighbour mask: pairs within cutoff, excluding diagonal
		N = d_true.shape[0]
		diag = torch.eye(N, dtype=torch.bool, device=d_true.device)
		neighbour = (d_true < cutoff) & ~diag  # (N, N)

		if neighbour.sum() == 0:
			return torch.tensor(0.0, device=t_b.device)

		delta = torch.abs(d_pred - d_true)  # (N, N)

		# Soft score per threshold: Cauchy kernel instead of hard indicator
		per_thresh = []
		for thr in thresholds:
			score = 1.0 / (1.0 + (delta / thr) ** 2)
			per_thresh.append(score[neighbour].mean())

		soft_lddt = torch.stack(per_thresh).mean()
		return 1.0 - soft_lddt  # loss form

	if batch is None:
		mask = None
		if plddt is not None:
			mask = (plddt.squeeze() >= plddt_thresh)
			if mask.sum() < 2:
				return torch.tensor(0.0, device=pred_t.device)
			return _lddt_single(pred_t[mask], true_coords[mask])
		return _lddt_single(pred_t, true_coords)

	losses = []
	for b in torch.unique(batch):
		idx = (batch == b).nonzero(as_tuple=True)[0]
		t_b = pred_t[idx]
		c_b = true_coords[idx]
		if plddt is not None:
			pmask = (plddt[idx].squeeze() >= plddt_thresh)
			if pmask.sum() < 2:
				continue
			t_b = t_b[pmask]
			c_b = c_b[pmask]
		if t_b.shape[0] < 2:
			continue
		losses.append(_lddt_single(t_b, c_b))

	if losses:
		return torch.stack(losses).mean()
	return torch.tensor(0.0, device=pred_t.device)


def compute_lddt_quaternions(pred_quats, pred_trans, target_coords, cutoff=15.0, thresholds=[0.5, 1.0, 2.0, 4.0]):
	"""
	Computes a local distance difference test (lDDT) score for a structure represented
	by a sequence of quaternion and translation transforms.
	
	Args:
		pred_quats (Tensor): Shape (N, 4) predicted unit quaternions.
		pred_trans (Tensor): Shape (N, 3) predicted translations.
		target_coords (Tensor): Shape (N, 3) ground-truth global coordinates.
		cutoff (float): Distance cutoff to consider neighbors.
		thresholds (list): Distance thresholds for lDDT scoring.
	
	Returns:
		A scalar tensor representing the lDDT score.
	"""
	# Compute predicted global coordinates from the quaternion chain.
	pred_coords = compute_chain_positions(pred_quats, pred_trans)
	N = pred_coords.shape[0]
	lddt_scores = []
	
	for i in range(N):
		# Compute distances from residue i to all others in the target structure.
		d_true = torch.norm(target_coords[i] - target_coords, dim=-1)
		# Consider only neighbors (and exclude self).
		neighbor_mask = (d_true < cutoff) & (torch.arange(N, device=target_coords.device) != i)
		if neighbor_mask.sum() == 0:
			continue  # No neighbors for this residue
		
		d_pred = torch.norm(pred_coords[i] - pred_coords, dim=-1)
		diff = torch.abs(d_pred - d_true)
		
		# For each threshold, determine the fraction of neighbor pairs with differences below the threshold.
		local_scores = []
		for thr in thresholds:
			local_scores.append((diff[neighbor_mask] < thr).float().mean())
		
		# Average the scores across thresholds for residue i.
		lddt_scores.append(torch.mean(torch.stack(local_scores)))
	
	return torch.mean(torch.stack(lddt_scores))


# ============================================================================
# Geometry Utility Functions
# ============================================================================

def rotation_matrix_to_quaternion(rot_matrices):
	"""
	Convert rotation matrices to quaternions (w, x, y, z) format.
	
	Args:
		rot_matrices (Tensor): Shape (*, 3, 3) rotation matrices
		
	Returns:
		Tensor: Shape (*, 4) quaternions in (w, x, y, z) format
	"""
	# Based on Shepperd's method for numerical stability
	# See: https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
	
	if rot_matrices.ndim == 2:
		rot_matrices = rot_matrices.unsqueeze(0)
		squeeze_output = True
	else:
		squeeze_output = False
	
	# Extract diagonal and off-diagonal elements
	r00 = rot_matrices[..., 0, 0]
	r11 = rot_matrices[..., 1, 1]
	r22 = rot_matrices[..., 2, 2]
	
	# Compute trace
	trace = r00 + r11 + r22
	
	# Initialize quaternion tensor
	quat = torch.zeros(*rot_matrices.shape[:-2], 4, dtype=rot_matrices.dtype, device=rot_matrices.device)
	
	# Case 1: trace > 0
	# clamp sqrt arguments to guard against tiny negative values from floating-point drift
	mask1 = trace > 0
	if mask1.any():
		s = torch.sqrt(torch.clamp(trace[mask1] + 1.0, min=0)) * 2  # s = 4*w
		quat[mask1, 0] = 0.25 * s  # w
		quat[mask1, 1] = (rot_matrices[mask1, 2, 1] - rot_matrices[mask1, 1, 2]) / s  # x
		quat[mask1, 2] = (rot_matrices[mask1, 0, 2] - rot_matrices[mask1, 2, 0]) / s  # y
		quat[mask1, 3] = (rot_matrices[mask1, 1, 0] - rot_matrices[mask1, 0, 1]) / s  # z
	
	# Case 2: r00 is largest diagonal
	mask2 = (~mask1) & (r00 > r11) & (r00 > r22)
	if mask2.any():
		s = torch.sqrt(torch.clamp(1.0 + r00[mask2] - r11[mask2] - r22[mask2], min=0)) * 2  # s = 4*x
		quat[mask2, 0] = (rot_matrices[mask2, 2, 1] - rot_matrices[mask2, 1, 2]) / s  # w
		quat[mask2, 1] = 0.25 * s  # x
		quat[mask2, 2] = (rot_matrices[mask2, 0, 1] + rot_matrices[mask2, 1, 0]) / s  # y
		quat[mask2, 3] = (rot_matrices[mask2, 0, 2] + rot_matrices[mask2, 2, 0]) / s  # z
	
	# Case 3: r11 is largest diagonal
	mask3 = (~mask1) & (~mask2) & (r11 > r22)
	if mask3.any():
		s = torch.sqrt(torch.clamp(1.0 + r11[mask3] - r00[mask3] - r22[mask3], min=0)) * 2  # s = 4*y
		quat[mask3, 0] = (rot_matrices[mask3, 0, 2] - rot_matrices[mask3, 2, 0]) / s  # w
		quat[mask3, 1] = (rot_matrices[mask3, 0, 1] + rot_matrices[mask3, 1, 0]) / s  # x
		quat[mask3, 2] = 0.25 * s  # y
		quat[mask3, 3] = (rot_matrices[mask3, 1, 2] + rot_matrices[mask3, 2, 1]) / s  # z
	
	# Case 4: r22 is largest diagonal
	mask4 = (~mask1) & (~mask2) & (~mask3)
	if mask4.any():
		s = torch.sqrt(torch.clamp(1.0 + r22[mask4] - r00[mask4] - r11[mask4], min=0)) * 2  # s = 4*z
		quat[mask4, 0] = (rot_matrices[mask4, 1, 0] - rot_matrices[mask4, 0, 1]) / s  # w
		quat[mask4, 1] = (rot_matrices[mask4, 0, 2] + rot_matrices[mask4, 2, 0]) / s  # x
		quat[mask4, 2] = (rot_matrices[mask4, 1, 2] + rot_matrices[mask4, 2, 1]) / s  # y
		quat[mask4, 3] = 0.25 * s  # z
	
	# Normalize quaternions — clamp denominator to avoid division by zero
	quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-8)
	
	if squeeze_output:
		quat = quat.squeeze(0)
	
	return quat


def reconstruct_positions(R, T):
	"""
	Reconstruct 3D CA positions from CA-to-CA displacement vectors.

	T[i] = CA[i+1] - CA[i] in the global frame, so positions are simply the
	cumulative sum of translations starting from the origin.  The rotation
	matrices R are kept as a parameter for API compatibility but are not used
	here — they represent the *local frame orientation*, not a transform that
	should be applied to global-frame displacements.

	Args:
		R (torch.Tensor): Local rotation matrices, shape (N, 3, 3)  [unused]
		T (torch.Tensor): CA-to-CA displacement vectors, shape (N, 3)

	Returns:
		torch.Tensor: Reconstructed CA positions of shape (N+1, 3), starting
		              from the origin.
	"""
	origin = torch.zeros(1, 3, dtype=T.dtype, device=T.device)
	return torch.cat([origin, torch.cumsum(T, dim=0)], dim=0)


