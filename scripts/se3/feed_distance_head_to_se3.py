"""SE3 refiner augmented with frozen distance-head pair predictions as features.

Instead of a weak auxiliary loss, this pipeline runs a frozen PairwiseDistanceHead
on each entry and concatenates its distance-bin predictions as *input features*
to the SE3 GotenNet refiner.  This gives the refiner direct access to learned
pair-distance statistics, much richer than the 90-dim frozen latents alone.

Use with the cache-only DDP trainer:
  python -m foldtree2.train_se3_lightning --config configs/se3_distance_head_input.yaml

Key change: ``forward()`` runs the distance head on the entry, takes its full
logits for a fixed set of pairs, and appends per-residue features to ``entry``
so the SE3 refiner receives them as additional node channels.
"""
import torch

from foldtree2.src.distance.distance_restraints import (
    PairwiseDistanceHead, eligible_pairs,
)
from foldtree2.train_se3_lightning import forward as se3_forward


def augment_features(distance_head, entry, device):
    """Run the distance head and append its predictions as node features.

    Returns an augmented entry with ``features`` expanded by:
    - distance-bin logits (num_bins) per pair, aggregated to per-residue via
      soft-contact aggregation (attention-weighted mean of contact probabilities).
    - The contact probability for each residue's neighborhood.
    """
    mask = entry['mask'].to(device)
    pairs = eligible_pairs(mask).to(device)
    node_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in entry.items()}
    with torch.no_grad():
        logits = distance_head(node_device, pairs)         # [P, num_bins]
        probs = logits.softmax(-1)                           # [P, num_bins]
        contact_probs = probs[:, :distance_head.contact_bins].sum(-1)  # [P]

    # Aggregate pair-level predictions back to per-residue features.
    # For each residue, average the distance distributions of all pairs it
    # participates in — a simple, fast aggregation without learned attention.
    n = len(mask)
    num_bins = probs.shape[-1]
    pair_features = torch.zeros(n, num_bins, device=device, dtype=entry['features'].dtype)
    pair_counts = torch.zeros(n, device=device, dtype=torch.long)
    i, j = pairs[:, 0], pairs[:, 1]
    pair_features.index_add_(0, i, probs.to(dtype=entry['features'].dtype))
    pair_features.index_add_(0, j, probs.to(dtype=entry['features'].dtype))
    pair_counts.index_add_(0, i, torch.ones(len(i), device=device, dtype=torch.long))
    pair_counts.index_add_(0, j, torch.ones(len(j), device=device, dtype=torch.long))
    pair_counts = pair_counts.clamp_min(1)
    pair_features = pair_features / pair_counts.unsqueeze(-1).float()
    pair_features = pair_features.masked_fill(~mask[:, None], 0.)

    # Concatenate: original features (40) + pair features (10)
    original_features = entry['features'].to(device)
    augmented = torch.cat([original_features, pair_features], dim=-1)

    result = dict(entry)
    result['features'] = augmented.cpu()
    return result


class DistanceAwareSE3Forward:
    """Wrapper that augments features with distance-head predictions.

    Drop-in for ``se3_validation.forward``.  The distance head is loaded
    once and used for every call.
    """
    def __init__(self, checkpoint_path, device):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.head = PairwiseDistanceHead(**state['model_config']).to(device)
        self.head.load_state_dict(state['model'])
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False
        print(f'Loaded frozen distance head from {checkpoint_path} '
              f'({sum(p.numel() for p in self.head.parameters())} params)')

    def __call__(self, model, entry, contact_policy='normal'):
        entry = augment_features(self.head, entry, next(model.parameters()).device)
        return se3_forward(model, entry, contact_policy)