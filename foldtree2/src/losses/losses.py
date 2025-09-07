import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import negative_sampling

EPS = 1e-8


def jensen_shannon_regularization(encodings):
    # 1) Compute the average distribution p
    p = encodings.mean(dim=0)
    
    # 2) Define uniform distribution u
    K = p.size(0)
    u = torch.ones_like(p) / K
    
    # 3) Compute the midpoint m = (p + u) / 2
    m = 0.5 * (p + u)
    
    # 4) Use the definition of JSD(p || u):
    # JSD(p || u) = 0.5 * KL(p || m) + 0.5 * KL(u || m)
    # KL(x || y) = sum( x_i * log(x_i / y_i) )
    eps = 1e-10
    
    kl_p_m = torch.sum(p * torch.log((p + eps) / (m + eps)))
    kl_u_m = torch.sum(u * torch.log((u + eps) / (m + eps)))
    
    jsd = 0.5 * kl_p_m + 0.5 * kl_u_m
    return jsd

#jaccard distance multiset loss for protein pairs

def jaccard_distance_multiset(A: torch.Tensor,
							  B: torch.Tensor,
							  dim: int = -1,
							  eps: float = 1e-8) -> torch.Tensor:
	"""
	Computes the generalized (multiset) Jaccard distance between two tensors A and B.
	Both A and B should be nonnegative and have the same shape.
	
	:param A: Tensor of shape (..., n_features)
	:param B: Tensor of shape (..., n_features)
	:param dim: Dimension along which to compute Jaccard. Default is the last dimension.
	:param eps: Small constant to avoid division by zero.
	:return: Tensor of Jaccard distances of shape (...).
	"""
	# Ensure A and B have the same shape
	if A.shape != B.shape:
		raise ValueError("A and B must have the same shape.")

	# Compute sum of minima and maxima along the chosen dimension
	min_sum = torch.minimum(A, B).sum(dim=dim)
	max_sum = torch.maximum(A, B).sum(dim=dim)

	# Compute Jaccard similarity
	jaccard_similarity = min_sum / (max_sum + eps)

	return jaccard_similarity


def recon_loss(data , pos_edge_index: Tensor , decoder = None , poslossmod = 1 , neglossmod= 1,  plddt = True  , offdiag = False , nclamp = 30 , key = None) -> Tensor:
	r"""Given latent variables :obj:`z`, computes the binary cross
	entropy loss for positive edges :obj:`pos_edge_index` and negative
	sampled edges.

	Args:
		z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
		pos_edge_index (torch.Tensor): The positive edges to train against.
		neg_edge_index (torch.Tensor, optional): The negative edges to
			train against. If not given, uses negative sampling to
			calculate negative edges. (default: :obj:`None`)
	"""
	#remove the diagonal
	pos_edge_index = pos_edge_index[:, pos_edge_index[0] != pos_edge_index[1]]
	res =decoder(data, pos_edge_index )

	if key == None:
		pos = res[1]
	if key != None:
		pos = res[key]
	
	#turn pos edge index into a binary matrix
	pos_loss = -torch.log( pos + EPS)
	if plddt == True:
		c1 = data['plddt'].x[pos_edge_index[0]].view(-1,1)
		c2 = data['plddt'].x[pos_edge_index[1]].view(-1,1)
		#both have to be above .5, binary and operation
		c1 = c1 > .5
		c2 = c2 > .5
		mask = c1 & c2
		pos_loss = pos_loss.view(-1,1)[ mask]
	if offdiag == True:
		#subtract the indices
		nres = pos_edge_index[0] - pos_edge_index[1]
		nres = torch.abs(nres)
		nres = torch.clamp(nres, max = nclamp)
		nres = nres / nclamp
		#sigmoid_modulation = torch.sigmoid(-(nres - nclamp / 2) / (nclamp / 10))
		pos_loss = pos_loss.view(-1,1) * nres.view(-1,1).float()
	
	pos_loss = pos_loss.mean()
	neg_edge_index = negative_sampling(pos_edge_index, data['res'].x.size(0))
	
	#remove the diagonal
	neg_edge_index = neg_edge_index[:, neg_edge_index[0] != neg_edge_index[1]]
	res = decoder(data ,  neg_edge_index )

	if key == None:
		neg = res[1]
	if key != None:
		neg = res[key]

	neg_loss = -torch.log( ( 1 - neg) + EPS )
	if plddt == False:
		c1 = data['plddt'].x[pos_edge_index[0]].view(-1,1)
		c2 = data['plddt'].x[pos_edge_index[1]].view(-1,1)
		#both have to be above .5, binary and operation
		c1 = c1 > .5
		c2 = c2 > .5
		mask = c1 & c2
		neg_loss = neg_loss.view(-1,1)[mask]
		
	if offdiag == True:
		#subtract the indices
		nres = neg_edge_index[0] - neg_edge_index[1]
		#take the absolute value
		nres = torch.abs(nres)
		nres = torch.clamp(nres, max = nclamp)
		#divide by nclamp
		nres = nres / nclamp
		#sigmoid_modulation = torch.sigmoid(-(nres - nclamp / 2) / (nclamp / 10))
		neg_loss = neg_loss.view(-1,1) * nres.view(-1,1).float()
	neg_loss = neg_loss.mean()
	return poslossmod*pos_loss + neglossmod*neg_loss  , torch.tensor(0.0)

def recon_loss_diag(data, pos_edge_index: Tensor, decoder=None, poslossmod=1, neglossmod=1, plddt=False, offdiag=False, nclamp=30, key=None) -> Tensor:
	# Remove the diagonal
	pos_edge_index = pos_edge_index[:, pos_edge_index[0] != pos_edge_index[1]]
	res = decoder(data, pos_edge_index)

	if key == None:
		pos = res[1]
	if key != None:
		pos = res[key]
	
	# Calculate distance from diagonal for positive edges
	diag_dist = torch.abs(pos_edge_index[0] - pos_edge_index[1]).float()
	# Normalize the distance weights to [1, 2] range - far edges get 2x weight
	pos_weights = 1.0 + (diag_dist / diag_dist.max())
	
	# Ensure consistent shapes for multiplication
	pos_loss = -torch.log(pos + EPS).squeeze()
	pos_loss = (pos_loss * pos_weights).unsqueeze(1)

	if 'edge_logits' in res and res['edge_logits'] is not None:
		#apply recon loss disto
		disto_loss_pos = recon_loss_disto(data, res, pos_edge_index, plddt=plddt, offdiag=offdiag, key='edge_logits') 

	if offdiag == True:
		nres = torch.abs(pos_edge_index[0] - pos_edge_index[1])
		nres = torch.clamp(nres, max=nclamp)
		nres = nres / nclamp
		pos_loss = (pos_loss.squeeze() * nres.float()).unsqueeze(1)
	if plddt == True:
		c1 = data['plddt'].x[pos_edge_index[0]].unsqueeze(1)
		c2 = data['plddt'].x[pos_edge_index[1]].unsqueeze(1)
		c1 = c1 > .5
		c2 = c2 > .5
		mask = c1 & c2
		mask = mask.squeeze(1)  # Ensure mask is 1D
		pos_loss = pos_loss[mask]
	pos_loss = pos_loss.mean()
	neg_edge_index = negative_sampling(pos_edge_index, data['res'].x.size(0))
	
	neg_edge_index = neg_edge_index[:, neg_edge_index[0] != neg_edge_index[1]]
	res = decoder(data, neg_edge_index)

	if key == None:
		neg = res[1]
	if key != None:
		neg = res[key]

	neg_loss = -torch.log((1 - neg) + EPS).squeeze()
	if offdiag == True:
		nres = torch.abs(neg_edge_index[0] - neg_edge_index[1])
		nres = torch.clamp(nres, max=nclamp)
		nres = nres / nclamp
		neg_loss = (neg_loss.squeeze() * nres.float()).unsqueeze(1)
	
	if plddt == True:
		c1 = data['plddt'].x[pos_edge_index[0]].unsqueeze(1)
		c2 = data['plddt'].x[pos_edge_index[1]].unsqueeze(1)
		c1 = c1 > .5
		c2 = c2 > .5
		mask = c1 & c2
		mask = mask.squeeze(1)  # Ensure mask is 1D	
		neg_loss = neg_loss[mask]
	
	neg_loss = neg_loss.mean()

	if 'edge_logits' in res and res['edge_logits'] is not None:
		#apply recon loss disto
		disto_loss_neg = recon_loss_disto(data, res, neg_edge_index, plddt=plddt, offdiag=offdiag, key='edge_logits' , no_bins=16) 

	return poslossmod*pos_loss + neglossmod*neg_loss, disto_loss_pos * poslossmod + disto_loss_neg * neglossmod

#amino acid onehot loss for x reconstruction
def aa_reconstruction_loss(x, recon_x):
	"""
	compute the loss over the node feature reconstruction.
	using categorical cross entropy
	"""
	x = torch.argmax(x, dim=1)
	#recon_x = torch.argmax(recon_x, dim=1)
	return F.cross_entropy(recon_x, x)

def gaussian_loss(mu , logvar , beta= 1.5):
	'''
	
	add beta to disentangle the features
	
	'''
	kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return beta*kl_loss


def recon_loss_disto(data , res , edge_index: Tensor,  plddt = True  , offdiag = False , nclamp = 30 ,no_bins = 16 , key = None) -> Tensor:

	'''
	Calculates a reconstruction loss based on predicted and true coordinates, with optional filtering by pLDDT confidence and off-diagonal weighting.

	Args:
		data (dict): Dictionary containing input features, including 'coords' and optionally 'plddt'.
		res (dict): Dictionary containing model outputs, indexed by `key`.
		edge_index (Tensor): Tensor of shape [2, num_edges] specifying pairs of indices for which to compute the loss.
		plddt (bool, optional): If True, only considers edges where both residues have pLDDT > 0.5. Default is True.
		offdiag (bool, optional): If True, weights the loss by the sequence separation between residue pairs. Default is False.
		nclamp (int, optional): Maximum sequence separation to clamp when weighting off-diagonal loss. Default is 30.
		key (str, optional): Key to select the relevant output from `res`.

	Returns:
		Tensor: Scalar tensor representing the mean reconstruction loss over selected edges.
	'''
	#remove the diagonal
	logits = res[key]
	#turn pos edge index into a binary matrix
	disto_loss = distogram_loss( logits, data['coords'].x , edge_index  , no_bins = no_bins)
	if plddt == True:
		c1 = data['plddt'].x[edge_index[0]].view(-1,1)
		c2 = data['plddt'].x[edge_index[1]].view(-1,1)
		#both have to be above .5, binary and operation
		c1 = c1 > .5
		c2 = c2 > .5
		mask = c1 & c2
		disto_loss = disto_loss.view(-1,1)[ mask]
	if offdiag == True:
		#subtract the indices
		nres = edge_index[0] - edge_index[1]
		nres = torch.abs(nres)
		nres = torch.clamp(nres, max = nclamp)
		nres = nres / nclamp
		disto_loss = disto_loss.view(-1,1) * nres.view(-1,1).float()
	disto_loss = disto_loss.mean()
	return disto_loss

def distogram_loss(
	logits,
	coords,
	edge_index,
	min_bin=2.3125,
	max_bin=21.6875,
	no_bins=16,
	eps=1e-6,
):
	"""
	Computes distogram loss for a set of logits and coordinates.
	Args:
		logits: (B, Npairs, no_bins) logits for samples
		coords: (B, N, 3) coordinates for samples
		edge_index: (2, Npairs) indices for pairs
		mask: (B, Npairs) mask for pairs (optional)
	Returns:
		Scalar loss averaged over batch
	"""
	boundaries = torch.linspace(
		min_bin,
		max_bin,
		no_bins - 1,
		device=logits.device,
	) ** 2  # (no_bins-1,)

	idx0, idx1 = edge_index[0], edge_index[1]
	dists = torch.sum(
		(coords[ idx0  , :] - coords[ idx1 , :]) ** 2,
		dim=-1,
		keepdim=True,
	)  # ( Npairs, 1)

	true_bins = torch.sum(dists > boundaries, dim=-1)  # (B, Npairs)

	errors = F.cross_entropy(
		logits,  # (Npairs, no_bins)
		true_bins,
		reduction="none",
	)  # (B, Npairs)

	return errors


#alphafold inspired losses using FAPE or LDDT with quaternions or rotation matrices
def fape_loss(true_R, true_t, pred_R, pred_t, batch, plddt= None, d_clamp=10.0, eps=1e-8 , temperature = .25 , reduction = 'mean' , soft = False):
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
		d_clamp (float, optional): Clamping threshold for error. (default: 10.0)
		eps (float, optional): Small constant for numerical stability. (default: 1e-8)
		
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
		local_pred = torch.einsum("mij,mnj->mni", pred_R[idx].transpose(1,2 ), diff_pred)
		
		# Compute pairwise differences for the true translations.
		diff_true = true_t[idx].unsqueeze(1) - true_t[idx].unsqueeze(0)  # shape: (m, m, 3)
		# Transform differences into the local true frames.
		local_true = torch.einsum("mij,mnj->mni", true_R[idx].transpose(1,2), diff_true)
		
		# Compute the L2 error per residue pair and clamp it.
		if soft == False:
			error = torch.norm(local_pred - local_true + eps, dim=-1)
			if plddt is not None:
				error = error*plddt[idx]
			error = torch.clamp(error, max=d_clamp)
			losses.append(error.mean())
		else:				
			# Compute pairwise squared Euclidean distances
			dist_sq = torch.cdist(local_pred, local_true, p=2).pow(2)
			dist_sq = torch.clamp(dist_sq, max=d_clamp**2)
			# Compute soft alignment probabilities using a Gaussian kernel
			soft_alignment = F.softmax(-dist_sq / temperature, dim=-1)
			# Compute soft FAPE loss
			weighted_distances = (soft_alignment * dist_sq).sum(dim=-1)  # (B, N)
			fape_loss = weighted_distances.mean() if reduction == 'mean' else weighted_distances.sum()
			if plddt is not None:
				fape_loss = fape_loss * plddt[idx]
			losses.append(fape_loss)
	if losses:
		return torch.stack(losses).mean()
	else:
		return torch.tensor(0.0, device=true_R.device)

def transform_rt_to_coordinates(rotations, translations):
	"""
	Convert R, t matrices into global 3D coordinates.
	"""
	batch_size, num_residues, _ = rotations.shape
	coords = torch.zeros((batch_size, num_residues, 3), device=rotations.device)
	for b in range(batch_size):
		transform = torch.eye(4, device=rotations.device)
		for i in range(num_residues):
			T = torch.eye(4, device=rotations.device)
			T[:3, :3] = rotations[b, i]
			T[:3, 3] = translations[b, i]
			transform = transform @ T  # Apply transformation
			coords[b, i] = transform[:3, 3]
	return coords

def quaternion_rotate(q, v):
	"""
	Rotates a batch of 3D points v by a unit quaternion q.
	
	Args:
		q: Tensor of shape (..., 4) representing unit quaternions in [w, x, y, z] format.
		v: Tensor of shape (..., 3) representing 3D points.
		
	Returns:
		Rotated 3D points of shape (..., 3).
	"""
	# Split the quaternion into its scalar and vector parts.
	q_w = q[..., 0:1]
	q_vec = q[..., 1:]
	
	# Use the efficient quaternion rotation formula:
	# v_rot = v + 2 * cross(q_vec, q_w * v + cross(q_vec, v))
	t = 2 * torch.cross(q_vec, v, dim=-1)
	v_rot = v + q_w * t + torch.cross(q_vec, t, dim=-1)
	return v_rot

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
	loss = torch.mean(torch.norm(points_pred - points_target, dim=-1))
	return loss

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


def compute_chain_positions(quaternions, translations, reference_coords=None):
	"""
	Apply rotation (quaternion) and translation to a set of 3D reference coordinates using PyTorch.
	
	Parameters:
	- quaternions: (N, 4) tensor of quaternions (x, y, z, w)
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
	
	x, y, z, w = quaternions.unbind(-1)
	
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
			# Apply current rotation to translation
			rotated_t = torch.matmul(global_R, translations[b,i])
			curr_pos = curr_pos + rotated_t
			chain_positions.append(curr_pos.clone())
			
			# Update cumulative rotation
			global_R = torch.matmul(global_R, rotations[b,i])

		positions.append(torch.stack(chain_positions))

	positions = torch.stack(positions)
	
	# Return to original shape if unbatched input
	if len(orig_shape) == 1:
		positions = positions.squeeze(0)
		
	return positions

def compute_lddt_loss(true_coords, pred_coords, cutoff=15.0):
	"""
	Compute lDDT loss for backpropagation.
	"""
	num_pairs = 0
	num_matching_pairs = 0
	true_dists = torch.cdist(true_coords, true_coords)  # (N, N)
	pred_dists = torch.cdist(pred_coords, pred_coords)  # (N, N)
	mask = (true_dists < cutoff).float()
	diff = torch.abs(true_dists - pred_dists)
	valid_pairs = (diff < 0.5 * true_dists) * mask
	num_pairs += torch.sum(mask)
	num_matching_pairs += torch.sum(valid_pairs)
	lddt_score = num_matching_pairs / num_pairs if num_pairs > 0 else 0
	return 1.0 - lddt_score  # Loss formulation

def lddt_loss(coord_true, pred_R, pred_t, batch, plddt= None, d_clamp=10.0, eps=1e-8 , reduction = 'mean' ):
	losses = []
	unique_batches = torch.unique(batch)
	for b in unique_batches:
		idx = (batch == b).nonzero(as_tuple=True)[0]
		coord_pred = transform_rt_to_coordinates(pred_R[idx], pred_t[idx])		
		lddt_loss = compute_lddt_loss(coord_true, coord_pred)
		if plddt is not None:
			lddt_loss = lddt_loss * plddt[idx].view(-1,1)
		losses.append(lddt_loss)
	if losses:
		return torch.stack(losses).mean()
	
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
			continue  # Skip if no neighbors are found.
		
		d_pred = torch.norm(pred_coords[i] - pred_coords, dim=-1)
		diff = torch.abs(d_pred - d_true)
		
		# For each threshold, determine the fraction of neighbor pairs with differences below the threshold.
		local_scores = []
		for thr in thresholds:
			local_scores.append((diff[neighbor_mask] < thr).float().mean())
		
		# Average the scores across thresholds for residue i.
		lddt_scores.append(torch.mean(torch.stack(local_scores)))
	
	return torch.mean(torch.stack(lddt_scores))



def quaternion_to_rotation_matrix(quat):
	"""
	Convert a batch of quaternions (x, y, z, w) into 3x3 rotation matrices.
	
	Parameters:
	- quat: (batch, N, 4) Tensor of quaternions (x, y, z, w)

	Returns:
	- rot_matrices: (batch, N, 3, 3) Tensor of rotation matrices
	"""
	assert quat.shape[-1] == 4, "Quaternions should have shape (*, 4)"
	
	norm = torch.norm(quat, dim=-1, keepdim=True)
	quat = quat / norm
	x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]


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
