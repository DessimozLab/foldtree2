import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math



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


def second_order_entropy_rate(q, alpha=1e-3, eps=1e-8):
    """
    q: [B, L, K] soft code assignments (sum_K q=1)
    returns: scalar H2 (estimated entropy rate)
    """
    B, L, K = q.shape
    if L < 3:
        return torch.tensor(0.0, device=q.device, dtype=q.dtype)

    q0 = q[:, :-2, :]   # [B, L-2, K]  z_{t-2}
    q1 = q[:, 1:-1, :]  # [B, L-2, K]  z_{t-1}
    q2 = q[:, 2:, :]    # [B, L-2, K]  z_t

    # Soft trigram counts C_{ij,k}: [K, K, K]
    C = torch.einsum("btk,btm,btn->kmn", q0, q1, q2)  # i=k, j=m, k=n (just naming)

    # Transition probs T_hat[i,j,k]
    C_smooth = C + alpha
    T_hat = C_smooth / (C_smooth.sum(dim=2, keepdim=True) + eps)  # normalize over next-state

    # Pair weights pi_hat[i,j] from soft pair counts
    C_pair = torch.einsum("btk,btn->kn", q[:, :-1, :], q[:, 1:, :])  # [K, K]
    pi_hat = C_pair / (C_pair.sum() + eps)

    # Conditional entropy per context (i,j): H(T_hat[i,j,*])
    H_cond = -(T_hat * (T_hat + eps).log()).sum(dim=2)  # [K, K]

    # Entropy rate: sum_{i,j} pi[i,j] * H_cond[i,j]
    H2 = (pi_hat * H_cond).sum()
    return H2


class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5, reset_threshold=100000, reset=False, klweight=1,
			  H2_weight = 0.1 , diversityweight=0, entropyweight=0, jsweight=0, prior_momentum=0.99, use_commitment_scheduling=False, commitment_warmup_steps=5000, commitment_schedule='cosine', commitment_start=0.1, commitment_end=None , **kwargs	):
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

		if self.H2_weight > 0:
			
			qindices = distances.log_softmax(dim=-1)  # [B*L]
			#add batch dimension back to qindices
			qindices = qindices.view(batch.size(0), -1)  # [B, L]
			#add batch dimension back to qindices
			qindices = qindices.view(batch.size(0), -1, self.num_embeddings)  # [B, L, K]
			H2 = second_order_entropy_rate(qindices )
			# Note: H2 is an estimate of the second-order entropy rate, which can
			# be used as a proxy for diversity and temporal structure in the code usage.
		# Total loss
		total_loss = loss \
			- self.entropyweight * entropy_reg \
			+ self.diversityweight * diversity_reg \
			+ self.klweight * kl_div_reg \
			- self.jsweight * jensen_shannon \
			+ self.H2_weight * H2

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

