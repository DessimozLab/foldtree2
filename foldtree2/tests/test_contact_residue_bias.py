import torch
import torch.nn.functional as F

from foldtree2.src.distance.contact_bias_experiment import ResidueBiasDistanceHead
from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, eligible_pairs
from foldtree2.tests.test_distance_restraints import example


def test_residue_bias_starts_at_baseline_and_learns_symmetrically():
    entry = example()
    pairs = eligible_pairs(entry['mask'])
    kwargs = dict(feature_dim=8, hidden=16, context_layers=2, spherical_dim=64)
    torch.manual_seed(42)
    baseline = PairwiseDistanceHead(**kwargs).eval()
    torch.manual_seed(42)
    variant = ResidueBiasDistanceHead(**kwargs).eval()
    torch.testing.assert_close(variant(entry, pairs), baseline(entry, pairs), atol=0, rtol=0)
    labels = torch.arange(len(pairs)) % 10
    F.cross_entropy(variant(entry, pairs), labels).backward()
    assert torch.isfinite(variant.residue_bias.weight.grad).all()
    assert variant.residue_bias.weight.grad.norm() > 0
    with torch.no_grad():
        variant.residue_bias.weight.copy_(variant.residue_bias.weight.grad * .1)
    torch.testing.assert_close(variant(entry, pairs), variant(entry, pairs.flip(-1)), atol=0, rtol=0)
    restored = ResidueBiasDistanceHead(**variant.model_config()).eval()
    restored.load_state_dict(variant.state_dict())
    torch.testing.assert_close(variant(entry, pairs), restored(entry, pairs), atol=0, rtol=0)
