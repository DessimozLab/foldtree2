import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


import torch
import torch.nn.functional as F

def soft_code_probs_from_distances(distances, tau=1.0, eps=1e-8):
    """
    distances: [N, K] (smaller = closer)
    returns:
      q:  [N, K] soft assignment probs
      p:  [K]    mean usage probs across N
    """
    # stable softmax over -dist/tau
    q = F.softmax(-distances / max(tau, eps), dim=-1)  # [N,K]
    p = q.mean(dim=0)                                  # [K]
    p = p.clamp_min(eps)
    p = p / p.sum()
    return q, p

def kl_to_target(p, target, eps=1e-8):
    """
    KL(p || target) with both as probs over K.
    """
    p = p.clamp_min(eps); p = p / p.sum()
    t = target.clamp_min(eps); t = t / t.sum()
    return torch.sum(p * (p.log() - t.log()))

def js_to_target(p, target, eps=1e-8):
    """
    Jensen-Shannon divergence (bounded, gentler than KL).
    """
    p = p.clamp_min(eps); p = p / p.sum()
    t = target.clamp_min(eps); t = t / t.sum()
    m = 0.5 * (p + t)
    return 0.5 * kl_to_target(p, m, eps) + 0.5 * kl_to_target(t, m, eps)

def laplace_smooth(p, alpha=1e-3):
    """
    Dirichlet/Laplace smoothing to reduce batch-noise in p.
    """
    K = p.numel()
    return (p + alpha) / (p.sum() + alpha * K)


def diversity_regularization_soft(q_NK=None, distances=None, tau=1.0, batch=None,
                                  alpha=1e-3, eps=1e-8, per_graph=True):
    """
    Diversity regularizer using SOFT assignments.

    Encourages the mean code-usage distribution p to be close to uniform:
        div = sum_k (p_k - 1/K)^2

    Inputs (choose one):
      - q_NK:      [N, K] soft assignment probabilities (rows sum to 1)
      - distances: [N, K] distances to codebook; will compute q_NK = softmax(-dist/tau)

    Optional:
      - batch: [N] graph ids (PyG). If provided and per_graph=True, compute per-graph usage then average
               (reduces length bias / variance).
      - alpha: Laplace smoothing strength on p (reduces batch noise)
    """
    if q_NK is None:
        assert distances is not None, "Provide q_NK or distances"
        q_NK = F.softmax(-distances / max(tau, eps), dim=-1)  # [N,K]

    N, K = q_NK.shape

    # Compute mean usage p
    if batch is not None and per_graph:
        # compact ids
        _, batch_c = torch.unique(batch, sorted=True, return_inverse=True)
        B = int(batch_c.max().item()) + 1

        sums = torch.zeros(B, K, device=q_NK.device, dtype=q_NK.dtype)
        counts = torch.zeros(B, 1, device=q_NK.device, dtype=q_NK.dtype)

        sums.index_add_(0, batch_c, q_NK)
        ones = torch.ones(N, 1, device=q_NK.device, dtype=q_NK.dtype)
        counts.index_add_(0, batch_c, ones)

        per_graph_p = sums / (counts + eps)      # [B,K]
        p = per_graph_p.mean(dim=0)              # [K]
    else:
        p = q_NK.mean(dim=0)                     # [K]

    # Smooth + normalize (in fp32 for safety)
    p = p.float()
    p = (p + alpha)
    p = p / (p.sum() + eps)

    # L2-to-uniform
    u = torch.full((K,), 1.0 / K, device=p.device, dtype=p.dtype)
    div = torch.sum((p - u) ** 2)
    return div




# Define the regularization functions outside the class
def entropy_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
	return entropy

def diversity_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	diversity_loss = torch.sum((probabilities - 1 / probabilities.size(0)) ** 2)
	return diversity_loss

def kl_divergence_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	kl_divergence = torch.sum(probabilities * torch.log(probabilities * probabilities.size(0) + 1e-10))
	return kl_divergence

def cosine_anneal(start, end, t, T):
	"""Cosine from start->end over T steps at step t (0-indexed)."""
	if T <= 0: return end
	c = 0.5 * (1 + math.cos(math.pi * min(max(t/T, 0.0), 1.0)))
	return end + (start - end) * c

def pyg_to_padded(q_NK, batch):
	"""
	Assumes nodes are in correct temporal order within each graph in the current tensor order.
	q_NK:  [N, K]
	batch: [N] graph id per node
	Returns:
	  q_pad: [B, Lmax, K]
	  mask:  [B, Lmax]
	"""
	device = q_NK.device
	N, K = q_NK.shape
	B = int(batch.max().item()) + 1

	# counts per graph
	counts = torch.bincount(batch, minlength=B)            # [B]
	Lmax = int(counts.max().item())

	# local time index within each graph, respecting current order
	# vectorized: local_t = arange within segments
	# build offsets per graph
	# Example trick: compute running counts per batch id
	local_t = torch.empty(N, device=device, dtype=torch.long)
	running = torch.zeros(B, device=device, dtype=torch.long)
	for n in range(N):
		b = int(batch[n].item())
		local_t[n] = running[b]
		running[b] += 1

	q_pad = torch.zeros((B, Lmax, K), device=device, dtype=q_NK.dtype)
	mask  = torch.zeros((B, Lmax), device=device, dtype=torch.bool)

	q_pad[batch, local_t] = q_NK
	mask[batch, local_t]  = True
	return q_pad, mask



def second_order_entropy_rate_masked(q, mask, alpha=1e-3, eps=1e-8):
	"""
	q:    [B, L, K] probabilities (sum_K q = 1) at valid positions
	mask: [B, L] True where positions are valid
	Returns scalar H2 = H(z_t | z_{t-1}, z_{t-2}) estimated from soft trigrams.
	"""
	B, L, K = q.shape
	if L < 3:
		return torch.tensor(0.0, device=q.device, dtype=q.dtype)

	# valid triple positions where t-2, t-1, t are all valid
	m0 = mask[:, :-2]
	m1 = mask[:, 1:-1]
	m2 = mask[:, 2:]
	m = (m0 & m1 & m2).float()  # [B, L-2]

	q0 = q[:, :-2, :]   # [B, L-2, K]
	q1 = q[:, 1:-1, :]
	q2 = q[:, 2:, :]

	# Soft trigram counts: C[i,j,k] = sum_{b,t} m[b,t]*q0[b,t,i]*q1[b,t,j]*q2[b,t,k]
	C = torch.einsum("bt, bti, btj, btk -> ijk", m, q0, q1, q2)  # [K,K,K]

	# Transition probs per context (i,j)
	C_s = C + alpha
	T = C_s / (C_s.sum(dim=2, keepdim=True) + eps)  # [K,K,K]

	# Pair weights pi[i,j] from valid pairs
	mp = (mask[:, :-1] & mask[:, 1:]).float()  # [B, L-1]
	qp0 = q[:, :-1, :]
	qp1 = q[:, 1:, :]
	C_pair = torch.einsum("bt, bti, btj -> ij", mp, qp0, qp1)  # [K,K]
	pi = C_pair / (C_pair.sum() + eps)

	H_cond = -(T * (T + eps).log()).sum(dim=2)   # [K,K]
	H2 = (pi * H_cond).sum()
	return H2

def pack_inorder(q_NK, batch):
	"""
	q_NK:  [N, K]
	batch: [N] contiguous 0..B-1
	returns:
	  q_pad: [B, Lmax, K]
	  mask:  [B, Lmax]
	"""
	device = q_NK.device
	N, K = q_NK.shape
	B = int(batch.max().item()) + 1

	counts = torch.bincount(batch, minlength=B)     # [B]
	Lmax = int(counts.max().item())

	local_t = torch.empty(N, device=device, dtype=torch.long)
	running = torch.zeros(B, device=device, dtype=torch.long)
	for n in range(N):
		b = int(batch[n].item())
		local_t[n] = running[b]
		running[b] += 1

	q_pad = torch.zeros((B, Lmax, K), device=device, dtype=q_NK.dtype)
	mask  = torch.zeros((B, Lmax), device=device, dtype=torch.bool)

	q_pad[batch, local_t] = q_NK
	mask[batch, local_t]  = True
	return q_pad, mask

def usage_entropy_from_qpad(q_pad, mask, eps=1e-8):
	w = mask.float().unsqueeze(-1)  # [B,L,1]
	p = (q_pad * w).sum(dim=(0,1)) / (w.sum() + eps)  # [K]
	p = p.clamp_min(eps)
	return -(p * p.log()).sum()


class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost,
			decay=0.99, 
			epsilon=1e-5, 
			reset_threshold=100000, 
			reset=False, 
			klweight= 0 ,
			H2_weight = 0.25,
			H2_tau = 0.1,
			diversityweight=0.5, 
			entropyweight=0, 
			jsweight=0.2,
			prior_momentum=0.95, 
			use_commitment_scheduling=False, 
			commitment_warmup_steps=5000, 
			commitment_schedule='cosine', commitment_start=0.1, commitment_end=None , **kwargs	):
		super(VectorQuantizerEMA, self).__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost_final = commitment_cost
		self.decay = decay
		self.epsilon = epsilon
		self.reset_threshold = reset_threshold
		self.reset = reset

		# Commitment cost scheduling parameters
		self.use_commitment_scheduling = use_commitment_scheduling
		self.commitment_warmup_steps = commitment_warmup_steps
		self.commitment_schedule = commitment_schedule  # 'cosine', 'linear', or 'none'
		self.commitment_start = commitment_start  # Starting commitment cost
		self.commitment_end = commitment_end if commitment_end is not None else commitment_cost
		self.current_step = 0
		# Set initial commitment cost based on whether scheduling is enabled
		if self.use_commitment_scheduling:
			self.commitment_cost = self.commitment_start  # Will be updated during training
		else:
			self.commitment_cost = commitment_cost  # Use constant value

		if kwargs.get('restart_interval') is not None:
			self.restart_interval = kwargs['restart_interval']
		else:
			self.restart_interval = commitment_warmup_steps // 5  # Default to 5 restarts

		# Regularization weights
		self.diversityweight = diversityweight
		self.klweight = klweight
		self.entropyweight = entropyweight
		self.jsweight = jsweight
		
		self.H2_weight = H2_weight
		self.H2_tau = H2_tau

		# Embeddings
		self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
		self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

		# EMA variables
		self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
		self.ema_w = nn.Parameter(self.embeddings.weight.clone())
		self.register_buffer('embedding_usage_count', torch.zeros(num_embeddings, dtype=torch.long))

		# Running average prior
		self.prior_momentum = prior_momentum
		self.register_buffer('running_prior', torch.ones(num_embeddings) / num_embeddings)

	def update_commitment_cost(self):
		"""
		Update commitment cost based on the current training step using the specified schedule.
		Called automatically during forward pass when training.
		"""
		if self.current_step >= self.commitment_warmup_steps:
			self.commitment_cost = self.commitment_end
			return
		
		t = self.current_step
		T = self.commitment_warmup_steps
		start = self.commitment_start
		end = self.commitment_end
		
		if t <= T:
			if self.commitment_schedule == 'cosine':
				# Cosine annealing from start to end
				c = 0.5 * (1 + math.cos(math.pi * t / T))
				self.commitment_cost = end + (start - end) * c
			elif self.commitment_schedule == 'linear':
				# Linear interpolation from start to end
				self.commitment_cost = start + (end - start) * (t / T)
			elif self.commitment_schedule == 'cosine_with_restart':
				# Cosine annealing with restarts
				if self.restart_interval is not None:
					restart_interval = self.restart_interval
				else:
					restart_interval = T // 5  # 5 restarts
				t_mod = t % restart_interval
				c = 0.5 * (1 + math.cos(math.pi * t_mod / restart_interval))
				self.commitment_cost = end + (start - end) * c
			else:
				# 'none' or any other value - use final value immediately
				self.commitment_cost = end
		else:  # 'none' or any other value - use final value immediately
			self.commitment_cost = end


	def forward(self, x, batch=None):
		# Update commitment cost schedule during training (only if scheduling is enabled)
		if self.training and self.use_commitment_scheduling:
			self.update_commitment_cost()
			self.current_step += 1

		flat_x = x.view(-1, self.embedding_dim)  # [N, D]

		# Distance and hard encoding (for actual quantization / EMA updates)
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					+ torch.sum(self.embeddings.weight**2, dim=1)
					- 2 * torch.matmul(flat_x, self.embeddings.weight.t()))  # [N, K]

		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [N,1]
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
		encodings.scatter_(1, encoding_indices, 1)  # [N,K] one-hot

		# Quantization (hard)
		quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

		# VQ losses
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# --------------------------
		# Gradient-friendly usage regularizers (compute usage from soft assignments)
		# --------------------------
		eps = 1e-8
		# Temperature for soft assignments used ONLY for regularizers
		tau_kl = getattr(self, "kl_tau", 1.0)

		# Soft assignments: q_soft = softmax(-dist/tau)
		q_soft = F.softmax(-distances / max(tau_kl, eps), dim=-1)  # [N,K]

		# Compute a stable usage distribution p_soft
		if batch is not None:
			# compact batch ids (safe even if batch not contiguous)
			_, batch_c = torch.unique(batch, sorted=True, return_inverse=True)  # [N] in 0..B-1

			# per-graph normalize then average graphs -> reduces length bias + variance
			B = int(batch_c.max().item()) + 1
			K = q_soft.size(1)
			sums = torch.zeros(B, K, device=q_soft.device, dtype=q_soft.dtype)
			counts = torch.zeros(B, 1, device=q_soft.device, dtype=q_soft.dtype)

			sums.index_add_(0, batch_c, q_soft)
			ones = torch.ones(q_soft.size(0), 1, device=q_soft.device, dtype=q_soft.dtype)
			counts.index_add_(0, batch_c, ones)

			per_graph = sums / (counts + eps)     # [B,K]
			p_soft = per_graph.mean(dim=0)        # [K]
		else:
			p_soft = q_soft.mean(dim=0)           # [K]

		# Optional Laplace / Dirichlet smoothing to reduce batch noise
		kl_alpha = getattr(self, "kl_alpha", 1e-3)
		p_soft = (p_soft + kl_alpha)
		p_soft = p_soft / (p_soft.sum() + eps)

		# For EMA prior tracking you might also want the hard usage:
		p_hard = encodings.mean(dim=0).clamp_min(eps)
		p_hard = p_hard / (p_hard.sum() + eps)

		# Regularization terms
		entropy_reg = 0.0
		diversity_reg = 0.0
		kl_div_reg = 0.0
		jensen_shannon = 0.0

		# Entropy/diversity: you can choose p_soft or p_hard; p_soft is smoother
		if self.entropyweight > 0:
			entropy_reg = -(p_soft * (p_soft + eps).log()).sum()

		if self.diversityweight > 0:
			K = p_soft.numel()
			tau_div = getattr(self, "div_tau", 1.0)
			div_reg = diversity_regularization_soft(distances=distances, tau=tau_div, batch=batch,
                                        alpha=getattr(self, "div_alpha", 1e-3))
			diversity_reg = div_reg
		
		# KL/JS to running prior (detach prior target for stability)
		if self.klweight > 0 or self.jsweight > 0:
			prior = self.running_prior.detach().clamp_min(eps)
			prior = prior / (prior.sum() + eps)

			if self.klweight > 0:
				# KL(p_soft || prior)
				kl_div_reg = torch.sum(p_soft * ((p_soft + eps).log() - (prior + eps).log()))

			if self.jsweight > 0:
				# JS(p_soft || prior) = 0.5 KL(p||m) + 0.5 KL(prior||m)
				m = 0.5 * (p_soft + prior)
				js1 = torch.sum(p_soft * ((p_soft + eps).log() - (m + eps).log()))
				js2 = torch.sum(prior  * ((prior  + eps).log() - (m + eps).log()))
				jensen_shannon = 0.5 * (js1 + js2)

		# --------------------------
		# Your H2 term (already soft + differentiable through distances)
		# --------------------------
		if self.H2_weight != 0:
			tau = getattr(self, "H2_tau", 1.0)
			q_NK = F.softmax(-distances / max(tau, eps), dim=-1)  # [N,K]

			# compact batch ids
			if batch is None:
				# if no batch, treat as single sequence
				batch_c = torch.zeros(q_NK.size(0), device=q_NK.device, dtype=torch.long)
			else:
				_, batch_c = torch.unique(batch, sorted=True, return_inverse=True)

			q_pad, mask = pack_inorder(q_NK, batch_c)
			H2 = second_order_entropy_rate_masked(q_pad, mask)
			Hloss = H2
		else:
			H2 = 0.0
			Hloss = 0.0

		# Total loss (same signs as your original intent)
		total_loss = loss \
			- self.entropyweight * entropy_reg \
			+ self.diversityweight * diversity_reg \
			+ self.klweight * kl_div_reg \
			- self.jsweight * jensen_shannon \
			+ self.H2_weight * Hloss

		# EMA updates (only during training)
		if self.training:
			with torch.no_grad():
				encodings_sum = encodings.sum(0)                 # [K]
				dw = torch.matmul(encodings.t(), flat_x)         # [K,D]

				# Update EMA cluster size / weights (hard assignments)
				self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
				self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

				# Normalize cluster sizes
				n = self.ema_cluster_size.sum()
				cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

				# Update embeddings
				normalized_weights = self.ema_w / cluster_size.unsqueeze(1)
				self.embeddings.weight.data.copy_(normalized_weights)

				# Update usage count
				self.embedding_usage_count.add_(encodings_sum.long())

				# Update running prior
				# Option A (recommended w/ KL-on-soft): track soft usage
				self.running_prior.mul_(self.prior_momentum).add_(p_soft, alpha=1 - self.prior_momentum)
				# Option B: track hard usage (old behavior)
				# self.running_prior.mul_(self.prior_momentum).add_(p_hard, alpha=1 - self.prior_momentum)

				self.running_prior.div_(self.running_prior.sum() + eps)

				if self.reset:
					self.reset_unused_embeddings()

		# Straight-through estimator
		quantized = x + (quantized - x).detach()
		return quantized, total_loss

	'''
	def forward(self, x , batch=None):
		# Update commitment cost schedule during training (only if scheduling is enabled)
		if self.training and self.use_commitment_scheduling:
			self.update_commitment_cost()
			self.current_step += 1
		
		flat_x = x.view(-1, self.embedding_dim)

		# Distance and encoding
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantization
		quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

		# Loss terms
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Regularization
		probabilities = encodings.mean(dim=0)
		if self.entropyweight > 0:
			entropy_reg = entropy_regularization(encodings)
		else:
			entropy_reg = 0
		if self.diversityweight > 0:
			diversity_reg = diversity_regularization(encodings)
		else:
			diversity_reg = 0
		if self.klweight > 0:
			kl_div_reg = torch.sum(probabilities * (torch.log(probabilities + 1e-10) - torch.log(self.running_prior + 1e-10)))
		else:
			kl_div_reg = 0
		jensen_shannon = 0  # optional
		if self.H2_weight != 0:
			tau = getattr(self, "H2_tau", 1.0)
			q_NK = torch.softmax(-distances / tau, dim=-1)  # [N,K]

			# compact batch ids
			_, batch_c = torch.unique(batch, sorted=True, return_inverse=True)  # [N] in 0..B-1

			# OPTIONAL but important: ensure grouped-by-batch ordering
			# If not grouped, sort by batch to make sequences contiguous (still preserves intra-graph order only if already contiguous)
			# Better is sort by (batch, pos) if you have pos.
			# print("batch grouped?", torch.all(batch_c[:-1] <= batch_c[1:]).item())

			q_pad, mask = pack_inorder(q_NK, batch_c)          # q_pad [B,L,K], mask [B,L]
			H2 = second_order_entropy_rate_masked(q_pad, mask) # scalar
			#H = usage_entropy_from_qpad(q_pad, mask)           # scalar
			Hloss = H2
		else:
			H2 = 0
			H = 0
			Hloss = 0
		
		# Total loss
		total_loss = loss \
			- self.entropyweight * entropy_reg \
			+ self.diversityweight * diversity_reg \
			+ self.klweight * kl_div_reg \
			- self.jsweight * jensen_shannon \
			+ self.H2_weight * Hloss

		# EMA updates (only during training)
		# Wrap in no_grad() to prevent these buffer updates from interfering with DDP gradient computation
		if self.training:
			with torch.no_grad():
				encodings_sum = encodings.sum(0)
				dw = torch.matmul(encodings.t(), flat_x)

				# Update EMA cluster size
				self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
				
				# Update EMA weights
				self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

				# Normalize cluster sizes
				n = self.ema_cluster_size.sum()
				cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
				
				# Update embeddings using copy_ to avoid DDP issues with direct assignment
				normalized_weights = self.ema_w / cluster_size.unsqueeze(1)
				self.embeddings.weight.data.copy_(normalized_weights)

				# Update usage count
				self.embedding_usage_count.add_(encodings_sum.long())

				# Update running prior
				self.running_prior.mul_(self.prior_momentum).add_(probabilities, alpha=1 - self.prior_momentum)
				self.running_prior.div_(self.running_prior.sum())

				if self.reset:
					self.reset_unused_embeddings()
		quantized = x + (quantized - x).detach()
		return quantized, total_loss
	'''
	def reset_unused_embeddings(self):
		"""
		Resets the embeddings that have not been used for a certain number of iterations.
		"""
		unused_embeddings = self.embedding_usage_count < self.reset_threshold
		num_resets = unused_embeddings.sum().item()
		if num_resets > 0:
			with torch.no_grad():
				# MEMORY OPTIMIZATION: Use copy_() instead of direct assignment to avoid DDP issues
				new_embeddings = torch.randn((num_resets, self.embedding_dim), device=self.embeddings.weight.device)
				self.embeddings.weight.data[unused_embeddings] = new_embeddings
			# Reset usage counts for the reset embeddings (in-place operation)
			self.embedding_usage_count[unused_embeddings] = 0

	def get_commitment_cost(self):
		"""
		Get the current commitment cost value.
		Useful for logging and monitoring during training.
		"""
		return self.commitment_cost
	
	def reset_commitment_schedule(self):
		"""
		Reset the commitment cost schedule to start from the beginning.
		Useful if you want to restart the warmup schedule during training.
		"""
		self.current_step = 0
		self.commitment_cost = self.commitment_start

	def discretize_z(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)
		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
		# Get the encoding that has the minimum distance
		closest_indices = torch.argmin(distances, dim=1)
		
		# Convert indices to characters
		char_list = [chr(idx.item()) for idx in closest_indices]
		return closest_indices, char_list

	def string_to_hex(self, s):
		# if string is ascii, convert to hex
		if all(ord(c) < 248 for c in s):
			return s.encode().hex()
		else:
			#throw an error
			raise ValueError('String contains non-ASCII characters')
		
	def string_to_embedding(self, s):
		
		# Convert characters back to indices
		indices = torch.tensor([ord(c) for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings
	
	def ord_to_embedding(self, s):
		# Convert characters back to indices
		indices = torch.tensor([c for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings

	def ord_to_embedding(self, s):
		# Convert characters back to indices
		indices = torch.tensor([c for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings

class VectorQuantizer(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost):
		super(VectorQuantizer, self).__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost

		self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
		self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

	def forward(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)

		# Calculate distances
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

		# Get the encoding that has the min distance
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantize the latents
		quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

		# Loss
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Straight-through estimator
		quantized = x + (quantized - x).detach()
		return quantized, loss

	def discretize_z(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)
		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
		# Get the encoding that has the minimum distance
		closest_indices = torch.argmin(distances, dim=1)
		
		# Convert indices to characters
		char_list = [chr(idx.item()) for idx in closest_indices]
		return closest_indices, char_list

	def string_to_hex(self, s):
		# if string is ascii, convert to hex
		if all(ord(c) < 248 for c in s):
			return s.encode().hex()
		else:
			#throw an error
			raise ValueError('String contains non-ASCII characters')
		
	def string_to_embedding(self, s):
		
		# Convert characters back to indices
		indices = torch.tensor([ord(c) for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		
		return embeddings

