import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import negative_sampling , batched_negative_sampling

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


'''
def recon_loss_diag(data, pos_edge_index: Tensor, decoder=None, poslossmod=1, neglossmod=1, plddt=False, nclamp=30, key=None , nbins=8 , plddt_thresh=0.3) -> Tensor:
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
	
	pos_loss = pos_loss.mean()
	neg_edge_index = batched_negative_sampling(pos_edge_index , data['res'].batch , force_undirected = True)
	
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

	neg_loss = neg_loss.mean()
	if 'edge_logits' in res and res['edge_logits'] is not None:
		#apply recon loss disto
		disto_loss_neg = recon_loss_disto(data, res, neg_edge_index, plddt=plddt, key='edge_logits' , no_bins=nbins , plddt_thresh=plddt_thresh)

	return poslossmod*pos_loss + neglossmod*neg_loss, disto_loss_pos * poslossmod + disto_loss_neg * neglossmod


'''


def recon_loss_diag(data, pos_edge_index: Tensor, decoder=None, poslossmod=1, neglossmod=1, plddt=False, nclamp=30, key=None , nbins=8 , plddt_thresh=0.3) -> Tensor:
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

	neg_loss = neg_loss.mean()		
	if 'edge_logits' in res and res['edge_logits'] is not None:
		#apply recon loss disto
		disto_loss_neg = recon_loss_disto(data, res, neg_edge_index, plddt=plddt, key='edge_logits' , no_bins=nbins , plddt_thresh=plddt_thresh)

	return poslossmod*pos_loss + neglossmod*neg_loss, disto_loss_pos * poslossmod + disto_loss_neg * neglossmod



def prody_reconstruction_loss(data, decoder=None, poslossmod=1, neglossmod=1, plddt=False,  nclamp=30, key=None , plddt_thresh=0.3) -> Tensor:
	for interaction_type in []:
		# Remove the diagonal
		pos_edge_index = data[f'{interaction_type}_edge_index']
		res = decoder(data, pos_edge_index)
		# Calculate distance from diagonal for positive edges
		# Normalize the distance weights to [1, 2] range - far edges get 2x weight
		# Ensure consistent shapes for multiplication
		pos_loss = -torch.log(pos + EPS).squeeze()
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
		neg_loss = -torch.log((1 - neg) + EPS).squeeze()

cce_loss = torch.nn.CrossEntropyLoss()
#amino acid onehot loss for x reconstruction
def aa_reconstruction_loss(x, recon_x):
	"""
	compute the loss over the node feature reconstruction.
	using categorical cross entropy
	"""
	return cce_loss(recon_x, x)

def ss_reconstruction_loss(ss, recon_ss, mask_plddt=False, plddt_threshold=0.3 , plddt_mask = None):
	"""
	compute the loss over the node feature reconstruction.
	using categorical cross entropy
	"""
	if mask_plddt:
		mask = (plddt_mask > plddt_threshold).squeeze()
		ss_loss = F.cross_entropy(recon_ss[mask], ss[mask])
	else:	
		ss_loss = F.cross_entropy(recon_ss, ss)
	return ss_loss

	
"""
def angles_reconstruction_loss(true, pred):
	delta = pred - true
	return (1.0 - torch.cos(delta)).mean()
"""

def angles_reconstruction_loss(true, pred, beta=0.5 , plddt_mask = None , plddt_thresh = 0.3):
	delta = torch.atan2(torch.sin(pred - true), torch.cos(pred - true))
	if plddt_mask is not None:
		mask = plddt_mask > plddt_thresh
		mask = mask.squeeze(1)  # Ensure mask is 1D
		delta = delta[mask]
	loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta)

	return loss.mean()


def gaussian_loss(mu , logvar , beta= 1.5):
	'''
	
	add beta to disentangle the features
	
	'''
	kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return beta*kl_loss


def recon_loss_disto(data , res , edge_index: Tensor,  plddt = True  , nclamp = 30 ,no_bins = 8 , key = None , plddt_thresh=0.3) -> Tensor:

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
	#min_bin=2.3125,
	#max_bin=21.6875,
	min_bin=5,
	max_bin=50,
	no_bins=8,
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

