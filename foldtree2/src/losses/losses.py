"""Loss Functions for FoldTree2 Structural Phylogenetics.

This module provides specialized loss functions for training FoldTree2 models,
including:

- Multi-task learning with gradient norm balancing
- Graph reconstruction losses for protein contact maps
- Amino acid sequence and secondary structure reconstruction
- Protein structure geometry losses (angles, coordinates, distograms)
- Regularization losses (Jensen-Shannon divergence, Jaccard distance)
- pLDDT-aware masking for quality filtering

The losses support PyTorch Geometric heterogeneous graphs and incorporate
structural biology domain knowledge for effective protein structure modeling.

Example:
	>>> # Contact map reconstruction with pLDDT masking
	>>> edge_loss, logit_loss = recon_loss_diag(
	...     data, edge_index, decoder, plddt=True, plddt_thresh=0.3
	... )
	>>> # Backbone angle reconstruction
	>>> angle_loss = angles_reconstruction_loss(
	...     true_angles, pred_angles, plddt_mask=plddt, plddt_thresh=0.3
	... )
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import negative_sampling , batched_negative_sampling

# Small epsilon value to prevent numerical instabilities (division by zero, log(0))
EPS = 1e-8





# ============================================================================
# Multi-Task Learning Utilities
# ============================================================================

class GradNormBalancer:
	"""Gradient Norm Balancer for Multi-Task Learning.
	
	Automatically balances loss weights across multiple tasks by analyzing gradient
	norms during training. This implements the GradNorm algorithm (Chen et al., 2018)
	which dynamically adjusts task weights to equalize training rates.
	
	The algorithm maintains learnable task weights that are optimized to match
	gradient norms to target values based on relative inverse training rates.
	This helps prevent tasks with large gradients from dominating training.
	
	Args:
		task_names (list[str]): Names of tasks to balance (e.g., ['sequence', 'contacts', 'angles']).
		alpha (float, optional): Hyperparameter controlling asymmetry. Default: 0.5.
			- alpha = 0: forces all tasks to train at same rate
			- alpha = 1.5: allows tasks to train at different rates
		lr_w (float, optional): Learning rate for task weight optimization. Default: 1e-3.
		device (str, optional): Device to place weights on. Default: 'cuda'.
	
	Attributes:
		task_names (list[str]): Task names for reference.
		T (int): Number of tasks.
		alpha (float): Asymmetry parameter.
		w (torch.nn.Parameter): Learnable task weights (T,), normalized to sum to T.
		opt_w (torch.optim.Adam): Optimizer for task weights.
		L0 (torch.Tensor): Initial loss values snapshot for computing relative rates.
	
	Example:
		>>> balancer = GradNormBalancer(['sequence', 'contacts', 'geometry'], alpha=1.0)
		>>> # During training:
		>>> losses = [seq_loss, contact_loss, geom_loss]
		>>> weights = balancer.update_weights(losses, model.encoder.parameters())
		>>> total_loss = sum(w * l for w, l in zip(weights, losses))
	
	Reference:
		Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018).
		GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks.
		In ICML.
	"""
	
	def __init__(self, task_names, alpha=0.5, lr_w=1e-3, device="cuda"):
		self.task_names = task_names
		self.T = len(task_names)  # Number of tasks
		self.alpha = alpha

		# Initialize learnable task weights (constrained to be positive and sum to T)
		self.w = torch.nn.Parameter(torch.ones(self.T, device=device))
		self.opt_w = torch.optim.Adam([self.w], lr=lr_w)

		self.L0 = None  # Snapshot of initial losses (set on first call to update_weights)

	@torch.no_grad()
	def _renorm_(self):
		"""Renormalize task weights to be positive and sum to T.
		
		This constraint ensures that on average, tasks maintain their original scale.
		Called after each optimizer step to enforce constraints.
		"""
		# Ensure all weights are positive (prevent negative weights)
		self.w.data.clamp_(min=1e-6)
		# Normalize so weights sum to T (maintains average scale)
		self.w.data *= (self.T / self.w.data.sum())



def update_weights(self, losses, shared_params):
		"""Update task weights based on gradient norms and training rates.
		
		This method implements the core GradNorm algorithm:
		1. Compute gradient norms for each weighted task loss w.r.t. shared parameters
		2. Calculate relative inverse training rates based on loss dynamics
		3. Set target gradient norms proportional to training rates^alpha
		4. Optimize weights to minimize difference between actual and target gradient norms
		
		Args:
			losses (list[torch.Tensor]): Unweighted scalar loss tensors [L1, L2, ...], one per task.
				Length must equal number of tasks (T).
			shared_params (iterable): Parameters to compute gradients with respect to.
				Typically the last layer(s) of a shared trunk/encoder.
		
		Returns:
			torch.Tensor: Detached task weights (T,), normalized to sum to T.
				These should be used to weight task losses in the total objective.
		
		Example:
			>>> # Get unweighted losses from model
			>>> losses = [sequence_loss, contact_loss, geometry_loss]
			>>> shared_params = model.encoder.output_layer.parameters()
			>>> weights = balancer.update_weights(losses, shared_params)
			>>> # Compute weighted total loss
			>>> total_loss = sum(w * l for w, l in zip(weights, losses))
		"""
		assert len(losses) == self.T, f"Expected {self.T} losses, got {len(losses)}"

		# Initialize L0 snapshot on first call (used to compute relative training rates)
		if self.L0 is None:
			with torch.no_grad():
				self.L0 = torch.tensor([l.item() for l in losses], device=self.w.device)

		# Compute gradient norms gi = || ∂(wi * Li) / ∂W ||_2 for each task i
		g = []
		for i, Li in enumerate(losses):
			wiLi = self.w[i] * Li
			# Compute gradients with respect to shared parameters
			grads = torch.autograd.grad(
				wiLi, shared_params, retain_graph=True, create_graph=True, allow_unused=True
			)
			# Filter out None gradients (parameters not connected to this task)
			grads = [gg for gg in grads if gg is not None]
			if len(grads) == 0:
				# No gradients for this task (disconnected)
				gi = torch.zeros([], device=self.w.device)
			else:
				# L2 norm of all gradients
				gi = torch.norm(torch.cat([gg.reshape(-1) for gg in grads]), p=2)
			g.append(gi)
		g = torch.stack(g)  # Shape: (T,)

		# Compute relative inverse training rates ri
		# ri measures how fast task i is training relative to others
		with torch.no_grad():
			L = torch.tensor([l.item() for l in losses], device=self.w.device)
			Li_ratio = L / (self.L0 + 1e-12)  # Current loss / initial loss
			ri = Li_ratio / (Li_ratio.mean() + 1e-12)  # Normalize to mean=1

		# Compute target gradient norms: g_bar * ri^alpha
		g_bar = g.detach().mean()  # Average gradient norm across tasks
		target = g_bar * (ri ** self.alpha)  # Shape: (T,)

		# GradNorm loss: minimize L1 distance between actual and target gradient norms
		gradnorm_loss = torch.sum(torch.abs(g - target))

		# Optimize task weights to minimize GradNorm loss
		self.opt_w.zero_grad(set_to_none=True)
		gradnorm_loss.backward()
		self.opt_w.step()
		self._renorm_()  # Enforce constraints (positive, sum to T)

		return self.w.detach()


class UncertaintyWeighting(torch.nn.Module):
	"""Uncertainty-based loss weighting (Kendall & Gal).
	
	Learns per-task log-variance parameters and computes:
		sum_i (0.5 * exp(-2 * log_sigma_i) * L_i + log_sigma_i)
	"""
	def __init__(self, task_names, init_log_sigma=0.0, device="cuda"):
		super().__init__()
		self.task_names = list(task_names)
		init = torch.full((len(self.task_names),), float(init_log_sigma))
		self.log_sigma = torch.nn.Parameter(init)
	
	def forward(self, losses, active=None):
		losses = torch.stack([
			l if torch.is_tensor(l) else torch.tensor(l, device=self.log_sigma.device, dtype=self.log_sigma.dtype)
			for l in losses
		]).to(device=self.log_sigma.device, dtype=self.log_sigma.dtype)
		if losses.numel() != self.log_sigma.numel():
			raise ValueError(f"Expected {self.log_sigma.numel()} losses, got {losses.numel()}")

		if active is None:
			active_mask = torch.ones_like(losses)
		else:
			active_mask = torch.zeros_like(losses)
			active = torch.as_tensor(active, device=losses.device, dtype=torch.long)
			if active.numel() > 0:
				active_mask[active] = 1.0

		precision = torch.exp(-2.0 * self.log_sigma)  # 1/sigma^2
		per_task = 0.5 * precision * losses + self.log_sigma
		return (active_mask * per_task).sum()

	def get_sigmas(self):
		return torch.exp(self.log_sigma.detach())



# ============================================================================
# Regularization Losses
# ============================================================================

def jensen_shannon_regularization(encodings):
	"""Compute Jensen-Shannon divergence between encoding distribution and uniform.
	
	This regularization encourages the empirical distribution of discrete encodings
	to be close to uniform, promoting balanced usage of codebook entries in VQ-VAE.
	Lower JSD means more uniform usage; higher JSD indicates some codes dominate.
	
	The Jensen-Shannon divergence is a symmetric and bounded (0 to log(2)) measure
	of similarity between distributions. It's defined as:
		JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
	where M = 0.5 * (P + Q) is the midpoint distribution.
	
	Args:
		encodings (torch.Tensor): One-hot encoded discrete assignments, shape (N, K)
			where N is batch size and K is number of codebook entries.
			Each row should sum to 1 (standard one-hot encoding).
	
	Returns:
		torch.Tensor: Scalar JSD value between empirical and uniform distributions.
			Lower is more uniform. Range: [0, log(2)] ≈ [0, 0.693].
	
	Example:
		>>> # VQ-VAE encodings: (batch_size, num_embeddings)
		>>> encodings = torch.zeros(100, 64).scatter_(1, indices.unsqueeze(1), 1)
		>>> jsd_loss = jensen_shannon_regularization(encodings)
		>>> # Add to total loss to encourage uniform codebook usage
		>>> total_loss = reconstruction_loss + 0.1 * jsd_loss
	"""
	# 1) Compute empirical average distribution from batch
	p = encodings.mean(dim=0)  # Shape: (K,), empirical usage frequencies
	
	# 2) Define uniform distribution over K categories
	K = p.size(0)
	u = torch.ones_like(p) / K  # Shape: (K,), equal probability for all codes
	
	# 3) Compute midpoint distribution (mixture of p and u)
	m = 0.5 * (p + u)  # Shape: (K,)
	
	# 4) Compute Jensen-Shannon divergence using KL divergence components
	# JSD(p || u) = 0.5 * KL(p || m) + 0.5 * KL(u || m)
	# where KL(x || y) = Σ x_i * log(x_i / y_i)
	eps = 1e-10  # Small constant to prevent log(0)
	
	# Compute KL(p || m)
	kl_p_m = torch.sum(p * torch.log((p + eps) / (m + eps)))
	# Compute KL(u || m)
	kl_u_m = torch.sum(u * torch.log((u + eps) / (m + eps)))
	
	# Final JSD is symmetric average of both KL terms
	jsd = 0.5 * kl_p_m + 0.5 * kl_u_m
	return jsd

def jaccard_distance_multiset(A: torch.Tensor,
							  B: torch.Tensor,
							  dim: int = -1,
							  eps: float = 1e-8) -> torch.Tensor:
	"""Compute generalized Jaccard similarity for multisets (allows multiplicities).
	
	The multiset Jaccard similarity generalizes the standard set-based Jaccard index
	to allow for repeated elements (multiplicities). For feature vectors A and B:
		J(A, B) = Σ min(A_i, B_i) / Σ max(A_i, B_i)
	
	This is useful for comparing protein feature profiles, residue compositions,
	or other count-based representations where elements can appear multiple times.
	Unlike L2 distance, Jaccard similarity is scale-invariant and bounded [0,1].
	
	Args:
		A (torch.Tensor): First tensor, shape (..., n_features).
			Must be non-negative (counts, frequencies, or normalized features).
		B (torch.Tensor): Second tensor, same shape as A.
			Must be non-negative.
		dim (int, optional): Dimension along which to compute Jaccard. Default: -1 (last dim).
		eps (float, optional): Small constant to prevent division by zero. Default: 1e-8.
	
	Returns:
		torch.Tensor: Jaccard similarity values, shape (...). Range: [0, 1].
			1.0 = identical multisets, 0.0 = completely disjoint.
	
	Raises:
		ValueError: If A and B have different shapes.
	
	Example:
		>>> # Compare amino acid composition profiles (20-dimensional)
		>>> protein_A = torch.tensor([5, 3, 0, 2, ...])  # Residue counts
		>>> protein_B = torch.tensor([4, 3, 1, 2, ...])
		>>> similarity = jaccard_distance_multiset(protein_A, protein_B)
		>>> # Use as a loss (minimize distance = maximize similarity)
		>>> loss = 1.0 - similarity  # Jaccard distance in [0, 1]
	
	Note:
		Despite the function name, this returns similarity (not distance).
		To get distance, use: distance = 1 - similarity
	"""
	# Validate inputs have same shape
	if A.shape != B.shape:
		raise ValueError(f"A and B must have the same shape. Got A: {A.shape}, B: {B.shape}")
	
	# Compute numerator: sum of element-wise minimums (intersection size for multisets)
	min_sum = torch.minimum(A, B).sum(dim=dim)
	
	# Compute denominator: sum of element-wise maximums (union size for multisets)
	max_sum = torch.maximum(A, B).sum(dim=dim)
	
	# Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
	jaccard_similarity = min_sum / (max_sum + eps)
	
	return jaccard_similarity

# ============================================================================
# Graph Reconstruction Losses
# ============================================================================

def recon_loss_diag(data, pos_edge_index: Tensor, decoder=None, poslossmod=1, neglossmod=1, plddt=False, nclamp=30, key=None , nbins=8 , plddt_thresh=0.3 , normalize=False) -> Tensor:
	"""Contact map reconstruction loss with diagonal removal and quality masking.
	
	This loss trains a decoder to predict protein contact maps (edges between residues)
	by comparing predicted probabilities against true positive and negative edges.
	It uses binary cross-entropy with:
	- Positive samples: true contacts from structural alignment
	- Negative samples: randomly sampled non-contacts (negative_sampling)
	- Optional pLDDT masking: only consider high-confidence regions
	- Optional distogram loss: binned distance predictions if edge_logits available
	
	The diagonal (self-contacts) is removed since residues always "contact" themselves.
	Negative sampling ensures balanced training and prevents trivial solutions.
	
	Args:
		data (HeteroData): PyTorch Geometric heterogeneous graph containing:
			- data['res'].x: residue features
			- data['res'].batch: batch assignment for each residue
			- data['plddt'].x: per-residue confidence scores (if plddt=True)
			- data['coords'].x: 3D coordinates (if using distogram loss)
		pos_edge_index (torch.Tensor): Ground truth contact edges, shape (2, num_edges).
			Each column is [source_idx, target_idx].
		decoder (nn.Module, optional): Decoder model with signature:
			decoder(data, edge_index) -> dict with 'edge_probs' and optionally 'edge_logits'.
		poslossmod (float, optional): Multiplier for positive edge loss. Default: 1.
		neglossmod (float, optional): Multiplier for negative edge loss. Default: 1.
		plddt (bool, optional): If True, mask out low-confidence residues. Default: False.
		nclamp (int, optional): Maximum sequence separation for distance weighting. Default: 30.
		key (str, optional): Key to extract edge probabilities from decoder output.
			If None, uses res[1]. If provided, uses res[key].
		nbins (int, optional): Number of bins for distogram loss. Default: 8.
		plddt_thresh (float, optional): Minimum pLDDT for masking. Default: 0.3.
		normalize (bool, optional): If True, normalize the loss by the number of residues. Default: False.
	Returns:
		tuple[torch.Tensor, torch.Tensor]: (edge_loss, distogram_loss)
			- edge_loss: Binary cross-entropy loss for edge existence prediction
			- distogram_loss: Binned distance prediction loss (0 if no edge_logits)
	
	Example:
		>>> # Standard usage without pLDDT masking
		>>> edge_loss, disto_loss = recon_loss_diag(
		...     data, contact_edges, decoder, plddt=False
		... )
		>>> total_loss = edge_loss + 0.1 * disto_loss
		>>>
		>>> # With pLDDT masking (only train on confident regions)
		>>> edge_loss, disto_loss = recon_loss_diag(
		...     data, contact_edges, decoder, plddt=True, plddt_thresh=0.5
		... )
	
	Note:
		The function performs negative sampling internally using PyTorch Geometric's
		batched_negative_sampling, which respects batch boundaries in batched graphs.
	"""
	# Remove self-loops (diagonal entries) from positive edges
	# Residues always "contact" themselves, so these are uninformative
	pos_edge_index = pos_edge_index[:, pos_edge_index[0] != pos_edge_index[1]]
	res = decoder(data, pos_edge_index)

	if key == None:
		pos = res[1]
	if key != None:
		pos = res[key]
	
	# Calculate distance from diagonal for positive edges
	diag_dist = torch.abs(pos_edge_index[0] - pos_edge_index[1]).float()
	# Normalize the distance weights to [1, 2] range - far edges get 2x weight
	# Ensure consistent shapes for multiplication
	pos_loss = -torch.log(pos + EPS).squeeze()
	
	if 'edge_logits' in res and res['edge_logits'] is not None:
		#apply recon loss disto
		disto_loss_pos = recon_loss_disto(data, res, pos_edge_index, plddt=plddt, key='edge_logits', no_bins=nbins , plddt_thresh=plddt_thresh) 

	if plddt == True:
		c1 = data['plddt'].x[pos_edge_index[0]].squeeze(1)
		c2 = data['plddt'].x[pos_edge_index[1]].squeeze(1)
		c1 = c1 > plddt_thresh
		c2 = c2 > plddt_thresh
		mask = c1 & c2
		mask = mask.squeeze(0)  # Ensure mask is 1D
		pos_loss = pos_loss[mask]
		pos_edge_index_filtered = pos_edge_index[:, mask]
	else:
		pos_edge_index_filtered = pos_edge_index
	
	pos_loss = pos_loss.mean()
	
	neg_edge_index = batched_negative_sampling(pos_edge_index, data['res'].batch , force_undirected = True)
	
	neg_edge_index = neg_edge_index[:, neg_edge_index[0] != neg_edge_index[1]]
	res = decoder(data, neg_edge_index)

	if key == None:
		neg = res[1]
	if key != None:
		neg = res[key]

	neg_loss = -torch.log((1 - neg) + EPS).squeeze()
	
	if plddt == True:
		c1 = data['plddt'].x[neg_edge_index[0]].squeeze(1)
		c2 = data['plddt'].x[neg_edge_index[1]].squeeze(1)
		c1 = c1 > plddt_thresh
		c2 = c2 > plddt_thresh
		mask = c1 & c2
		mask = mask.squeeze(0)  # Ensure mask is 1D	
		neg_loss = neg_loss[mask]
		neg_edge_index_filtered = neg_edge_index[:, mask]
	else:
		neg_edge_index_filtered = neg_edge_index
	

	if 'edge_logits' in res and res['edge_logits'] is not None:
		#apply recon loss disto
		disto_loss_neg = recon_loss_disto(data, res, neg_edge_index, plddt=plddt, key='edge_logits' , no_bins=nbins , plddt_thresh=plddt_thresh)


	return poslossmod*pos_loss.mean() + neglossmod*neg_loss.mean(), disto_loss_pos.mean() * poslossmod + disto_loss_neg.mean() * neglossmod



def prody_reconstruction_loss(data, decoder=None, poslossmod=1, neglossmod=1, plddt=False,  nclamp=30, key=None , plddt_thresh=0.3) -> Tensor:
	for interaction_type in []:
		# Remove the diagonal
		pos_edge_index = data[f'{interaction_type}_edge_index']
		res = decoder(data, pos_edge_index)
		# Calculate distance from diagonal for positive edges
		# Normalize the distance weights to [1, 2] range - far edges get 2x weight
		# Ensure consistent shapes for multiplication
		#pos_loss = -torch.log(pos + EPS).squeeze()
		pos_loss = F.binary_cross_entropy_with_logits( pos , torch.ones_like(pos) )

		if offdiag == True:
			nres = torch.abs(pos_edge_index[0] - pos_edge_index[1])
			nres = torch.clamp(nres, max=nclamp)
			nres = nres / nclamp
			pos_loss = (pos_loss.squeeze() * nres.float()).unsqueeze(1)
		if plddt == True:
			c1 = data['plddt'].x[pos_edge_index[0]].unsqueeze(1)
			c2 = data['plddt'].x[pos_edge_index[1]].unsqueeze(1)
			c1 = c1 > .3
			c2 = c2 > .3
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

		#bce with logits
		neg_loss = F.binary_cross_entropy_with_logits( neg , torch.zeros_like(neg) )
		#neg_loss = -torch.log((1 - neg) + EPS).squeeze()

# ============================================================================
# Sequence and Structure Reconstruction Losses
# ============================================================================

# Cross-entropy loss for categorical predictions (amino acid types)

def aa_reconstruction_loss(x, recon_x , normalize = False):
	"""Compute amino acid sequence reconstruction loss.
	
	This loss trains the decoder to reconstruct amino acid identities from
	latent structural encodings. It uses cross-entropy between predicted
	logits and ground truth one-hot amino acid labels.
	
	Args:
		x (torch.Tensor): Ground truth amino acid one-hot encodings, shape (N, 20).
			Each row should be a one-hot vector for one of the 20 standard amino acids.
		recon_x (torch.Tensor): Predicted amino acid logits (unnormalized), shape (N, 20).
			Outputs from the decoder before softmax.
	
	Returns:
		torch.Tensor: Scalar cross-entropy loss averaged over all residues.
	
	Example:
		>>> # Ground truth one-hot (e.g., Alanine at position 0)
		>>> true_aa = torch.zeros(100, 20)
		>>> true_aa[0, 0] = 1  # First position is Alanine
		>>> # Model predictions (logits)
		>>> pred_logits = decoder(latent_z)
		>>> loss = aa_reconstruction_loss(true_aa, pred_logits)
	
	Note:
		The function expects one-hot encoded targets (not class indices).
		Cross-entropy will internally convert these to class indices.
	"""

	return F.cross_entropy(recon_x, x)


def ss_reconstruction_loss(ss, recon_ss, mask_plddt=False, plddt_threshold=0.3 , plddt_mask = None , normalize = False):
	"""Compute secondary structure reconstruction loss with optional quality masking.
	
	This loss trains the decoder to predict protein secondary structure (helix, sheet, coil)
	from latent structural encodings using 3-class cross-entropy. Optional pLDDT masking
	allows focusing on high-confidence regions where secondary structure is well-defined.
	
	Args:
		ss (torch.Tensor): Ground truth secondary structure labels, shape (N,).
			Integer class labels: 0=coil, 1=helix, 2=sheet (DSSP convention).
		recon_ss (torch.Tensor): Predicted secondary structure logits, shape (N, 3).
			Unnormalized scores for each of 3 classes per residue.
		mask_plddt (bool, optional): If True, only compute loss on high-confidence residues.
			Default: False.
		plddt_threshold (float, optional): Minimum pLDDT confidence for masking. Default: 0.3.
		plddt_mask (torch.Tensor, optional): Per-residue pLDDT confidence scores, shape (N, 1).
			Required if mask_plddt=True.
	
	Returns:
		torch.Tensor: Scalar cross-entropy loss, averaged over selected residues.
			Returns 0 if no residues pass pLDDT threshold (prevents NaN).
	
	Example:
		>>> # Without pLDDT masking (train on all residues)
		>>> ss_loss = ss_reconstruction_loss(true_ss, pred_ss, mask_plddt=False)
		>>>
		>>> # With pLDDT masking (only train on confident regions)
		>>> ss_loss = ss_reconstruction_loss(
		...     true_ss, pred_ss, mask_plddt=True,
		...     plddt_threshold=0.5, plddt_mask=plddt_scores
		... )
	
	Note:
		Secondary structure is assigned by DSSP (pydssp) during data preprocessing.
		Masking by pLDDT is recommended for AlphaFold structures where low-confidence
		regions may have unreliable secondary structure assignments.
	"""
	if mask_plddt:
		# Create boolean mask for high-confidence residues
		mask = (plddt_mask > plddt_threshold).squeeze()
		if mask.sum() > 0:
			# Compute loss only on masked residues
			ss_loss = F.cross_entropy(recon_ss[mask], ss[mask])
		else:
			# No residues pass threshold - return zero loss to prevent NaN
			ss_loss = torch.tensor(0.0, device=recon_ss.device)
	else:
		# Compute loss on all residues
		ss_loss = F.cross_entropy(recon_ss, ss)
	return ss_loss

	
def angles_reconstruction_loss(true, pred, beta=0.5 , plddt_mask = None , plddt_thresh = 0.3 , normalize = False):
	"""Compute backbone dihedral angle reconstruction loss with circular distance.
	
	This loss trains the decoder to predict protein backbone torsion angles (phi, psi, omega)
	from latent encodings. It correctly handles the circular nature of angles using:
		delta = atan2(sin(pred - true), cos(pred - true))
	which wraps angular differences to [-π, π] to avoid discontinuities at ±180°.
	
	Uses smooth L1 loss (Huber loss) which is less sensitive to outliers than L2.
	Optional pLDDT masking focuses training on high-confidence regions.
	
	Args:
		true (torch.Tensor): Ground truth angles in radians, shape (N, 3).
			Columns are typically [phi, psi, omega] backbone dihedrals.
		pred (torch.Tensor): Predicted angles in radians, shape (N, 3).
			Should be in same order/range as true angles.
		beta (float, optional): Threshold for smooth L1 loss transition. Default: 0.5.
			Smaller beta = more robust to outliers, larger = closer to L2.
		plddt_mask (torch.Tensor, optional): Per-residue confidence scores, shape (N, 1).
			If provided, filters to high-confidence residues.
		plddt_thresh (float, optional): Minimum pLDDT for masking. Default: 0.3.
	
	Returns:
		torch.Tensor: Scalar smooth L1 loss on circular angular differences.
			Returns 0 if no residues pass pLDDT threshold.
	
	Example:
		>>> # Predict backbone angles (phi, psi, omega)
		>>> true_angles = data['bondangles'].x  # (N, 3) from PDB
		>>> pred_angles = decoder(latent_z)      # (N, 3) predictions
		>>> loss = angles_reconstruction_loss(true_angles, pred_angles)
		>>>
		>>> # With pLDDT masking (recommended for AlphaFold structures)
		>>> loss = angles_reconstruction_loss(
		...     true_angles, pred_angles,
		...     plddt_mask=plddt_scores, plddt_thresh=0.5
		... )
	
	Note:
		Angles are computed from PDB coordinates during preprocessing using
		BioPython's calc_dihedral function. They represent protein backbone geometry.
		Circular distance is essential because 179° and -179° are actually very close!
	
	Reference:
		Smooth L1 (Huber) loss: Girshick, R. (2015). Fast R-CNN. ICCV.
	"""
	# Compute circular angular difference in [-π, π]
	# atan2 correctly handles the wrap-around at ±180°
	delta = torch.atan2(torch.sin(pred - true), torch.cos(pred - true))
	
	# Apply pLDDT masking if requested
	if plddt_mask is not None:
		mask = plddt_mask > plddt_thresh
		mask = mask.squeeze(1)  # Ensure mask is 1D for indexing
		if mask.sum() > 0:
			delta = delta[mask]  # Keep only high-confidence residues
		else:
			# No residues pass threshold - return zero to prevent NaN
			return torch.tensor(0.0, device=pred.device)
	
	# Smooth L1 loss (Huber loss) - robust to outliers
	# Target is zero since delta already represents the error
	loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta)
	return loss


def gaussian_loss(mu , logvar , beta= 1.5):
	"""Compute KL divergence loss for variational autoencoders (VAE).
	
	This loss regularizes the latent space of a VAE to follow a standard Gaussian
	distribution N(0, I). It measures the KL divergence between the learned latent
	distribution q(z|x) ~ N(mu, exp(logvar)) and the prior p(z) ~ N(0, I):
	
		KL[q(z|x) || p(z)] = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
	
	The beta parameter controls the weight of this regularization, enabling
	β-VAE disentanglement where higher beta encourages more independent latent factors.
	
	Args:
		mu (torch.Tensor): Mean of latent distribution, shape (batch_size, latent_dim).
		logvar (torch.Tensor): Log variance of latent distribution, same shape as mu.
			Using log(σ²) instead of σ² improves numerical stability.
		beta (float, optional): Weight for KL term (β-VAE). Default: 1.5.
			- beta = 1.0: standard VAE
			- beta > 1.0: β-VAE, encourages disentangled representations
			- beta < 1.0: less regularization, prioritizes reconstruction
	
	Returns:
		torch.Tensor: Scalar weighted KL divergence loss.
	
	Example:
		>>> # VAE encoder outputs
		>>> mu, logvar = encoder(x)  # Both (batch_size, latent_dim)
		>>> # Sample latent: z = mu + eps * exp(0.5 * logvar)
		>>> z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
		>>> # Compute losses
		>>> recon_loss = F.mse_loss(decoder(z), x)
		>>> kl_loss = gaussian_loss(mu, logvar, beta=2.0)  # β-VAE
		>>> total_loss = recon_loss + kl_loss
	
	Reference:
		Kingma & Welling (2013). Auto-Encoding Variational Bayes. ICLR.
		Highins et al. (2017). β-VAE: Learning Basic Visual Concepts. ICLR.
	"""
	# KL divergence: KL[N(mu, exp(logvar)) || N(0, I)]
	# Analytical form for Gaussian distributions
	kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	
	# Apply beta weighting for β-VAE disentanglement
	return beta * kl_loss


# ============================================================================
# Geometric/Distance-Based Losses
# ============================================================================

def recon_loss_disto(data , res , edge_index: Tensor,  plddt = True  , nclamp = 30 ,no_bins = 8 , key = None , plddt_thresh=0.3) -> Tensor:
	"""Compute distance distribution (distogram) reconstruction loss for protein structure.
	
	This loss trains the model to predict pairwise inter-residue distances by discretizing
	the continuous distance space into bins and using cross-entropy. This is more informative
	than binary contact prediction and helps capture longer-range structural information.
	
	The distogram approach (popularized by AlphaFold) converts distance prediction into
	a multi-class classification problem, making it more stable and easier to optimize
	than direct coordinate regression.
	
	Args:
		data (HeteroData): Graph data containing:
			- data['coords'].x: True 3D coordinates, shape (N, 3)
			- data['plddt'].x: Confidence scores, shape (N, 1) (if plddt=True)
		res (dict): Decoder outputs containing:
			- res[key]: Distance bin logits, shape (num_edges, no_bins)
		edge_index (torch.Tensor): Residue pairs to evaluate, shape (2, num_edges).
		plddt (bool, optional): If True, mask out low-confidence residues. Default: True.
		nclamp (int, optional): Maximum sequence separation for distance weighting. Default: 30.
		no_bins (int, optional): Number of distance bins. Default: 8.
			Bins are spaced from min_bin² to max_bin² (see distogram_loss).
		key (str, optional): Key to extract distance logits from res. Default: None.
		plddt_thresh (float, optional): Minimum pLDDT for masking. Default: 0.3.
	
	Returns:
		torch.Tensor: Scalar cross-entropy loss averaged over selected edges.
	
	Example:
		>>> # Predict binned distances for protein contacts
		>>> decoder_output = decoder(data, edge_index)
		>>> disto_loss = recon_loss_disto(
		...     data, decoder_output, edge_index,
		...     plddt=True, no_bins=16, key='edge_logits'
		... )
		>>> # Combine with contact loss
		>>> total_loss = contact_loss + 0.25 * disto_loss
	
	Note:
		Distances are computed in Euclidean space (Å units) from C-alpha coordinates.
		Binning strategy follows AlphaFold: squared distance bins for better resolution
		at short ranges where contacts matter most.
	
	Reference:
		Senior et al. (2020). Improved protein structure prediction using potentials
		from deep learning. Nature.
	"""
	# Extract predicted distance bin logits from decoder output
	logits = res[key]
	#turn pos edge index into a binary matrix
	disto_loss = distogram_loss( logits, data['coords'].x , edge_index  , no_bins = no_bins)
	if plddt == True:
		c1 = data['plddt'].x[edge_index[0]].view(-1,1)
		c2 = data['plddt'].x[edge_index[1]].view(-1,1)
		#both have to be above .5, binary and operation
		c1 = c1 > plddt_thresh
		c2 = c2 > plddt_thresh
		mask = c1 & c2
		mask = mask.squeeze(1)  # Ensure mask is 1D
		disto_loss = disto_loss[mask]
	disto_loss = disto_loss.mean()
	return disto_loss

def distogram_loss(
	logits,
	coords,
	edge_index,
	#min_bin=2.3125,  # AlphaFold default min (commented)
	#max_bin=21.6875,  # AlphaFold default max (commented)
	min_bin=2,         # Current: start at 5 Å
	max_bin=21,        # Current: end at 50 Å  
	no_bins=8,         # Number of distance bins
	eps=1e-6,          # Numerical stability epsilon
):
	"""Compute cross-entropy loss for binned distance predictions (distogram).
	
	This function converts continuous distance prediction into a multi-class classification
	problem by discretizing the distance range into bins. It computes the true bin for each
	residue pair from coordinates, then calculates cross-entropy against predicted bins.
	
	Bins are defined on SQUARED distances for better resolution at short ranges:
		- Bin boundaries: [min_bin², ..., max_bin²] with (no_bins - 1) boundaries
		- Bin assignment: bin_i if boundary[i-1] < d² ≤ boundary[i]
		- Bins below min: bin 0, bins above max: bin (no_bins - 1)
	
	This binning strategy prioritizes accuracy at short distances where contacts form.
	
	Args:
		logits (torch.Tensor): Predicted distance bin logits, shape (num_edges, no_bins).
			Unnormalized scores for each bin (before softmax).
		coords (torch.Tensor): 3D coordinates (typically C-alpha), shape (N, 3).
			In Angstroms. Usually from data['coords'].x.
		edge_index (torch.Tensor): Residue pairs to evaluate, shape (2, num_edges).
			edge_index[0] = source indices, edge_index[1] = target indices.
		min_bin (float, optional): Minimum distance for binning (Å). Default: 5.
			Distances below this go to bin 0.
		max_bin (float, optional): Maximum distance for binning (Å). Default: 50.
			Distances above this go to bin (no_bins - 1).
		no_bins (int, optional): Number of distance bins. Default: 8.
			More bins = finer resolution but harder to learn.
		eps (float, optional): Small constant for numerical stability. Default: 1e-6.
	
	Returns:
		torch.Tensor: Per-edge cross-entropy errors, shape (num_edges,).
			NOT reduced to scalar - allows for downstream masking/weighting.
	
	Example:
		>>> # Predict distance bins for protein structure
		>>> coords = data['coords'].x  # (num_residues, 3)
		>>> edge_index = contact_pairs  # (2, num_pairs)
		>>> logits = model.predict_distances(features)  # (num_pairs, 16)
		>>> errors = distogram_loss(
		...     logits, coords, edge_index, no_bins=16, min_bin=3, max_bin=20
		... )
		>>> # Apply pLDDT masking before reduction
		>>> loss = errors[high_confidence_mask].mean()
	
	Note:
		- AlphaFold uses squared bins: min_bin=2.3125, max_bin=21.6875, no_bins=64
		- Current defaults (5-50 Å, 8 bins) are more coarse but faster to train
		- Returns unreduced loss to allow flexible masking strategies
	"""
	# Define bin boundaries on squared distances
	# Squaring gives better resolution at short ranges (e.g., 5²=25, 50²=2500)
	boundaries = torch.linspace(
		min_bin,
		max_bin,
		no_bins - 1,  # (no_bins - 1) boundaries define (no_bins) bins
		device=logits.device,
	) ** 2  # Shape: (no_bins - 1,)

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

