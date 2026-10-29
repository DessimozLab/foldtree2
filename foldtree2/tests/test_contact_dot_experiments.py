import pytest
import torch
import torch.nn.functional as F

from foldtree2.src.distance.contact_dot_experiments import PairAdjustedDotHead, SymmetricCrossDotHead
from foldtree2.src.distance.contact_cross_dot_perturbed import PerturbedCrossDotHead
from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, eligible_pairs
from foldtree2.tests.test_distance_restraints import example
from scripts.contact.export_contact_operating_point import shifted_checkpoint


def test_global_logit_shift_changes_cutoff_but_preserves_ranking():
    probabilities = torch.tensor([.02, .08, .14, .21, .31], dtype=torch.float64)
    threshold = torch.tensor(.15, dtype=torch.float64)
    adjusted = torch.sigmoid(torch.logit(probabilities) - torch.logit(threshold))
    assert torch.equal(adjusted >= .5, probabilities >= threshold)
    assert torch.equal(adjusted.argsort(), probabilities.argsort())


def test_exported_bias_shift_preserves_original_and_changes_the_cutoff():
    import math
    threshold = .2
    checkpoint = {'config': {'spherical_dim': 64}, 'state_dict': {'contact_offset': torch.tensor(-3.)}}
    calibration = {'calibration': {'threshold': threshold},
                   'logit_bias_shift': -math.log(threshold / (1 - threshold)),
                   'heldout': {'precision': .1, 'recall': .2}}
    shifted = shifted_checkpoint(checkpoint, calibration)
    assert checkpoint['state_dict']['contact_offset'].item() == -3.
    assert shifted['state_dict']['contact_offset'].item() == pytest.approx(-3. + calibration['logit_bias_shift'])
    assert 'not calibrated probabilities' in shifted['operating_point']['meaning']
    assert 'distance-bin' in shifted['operating_point']['effect']


def test_perturbed_cross_dot_breaks_symmetry_without_changing_pair_order():
    entry = example()
    pairs = eligible_pairs(entry['mask'])
    config = dict(feature_dim=8, hidden=16, context_layers=2, spherical_dim=64)
    torch.manual_seed(42)
    baseline = PairwiseDistanceHead(**config).eval()
    torch.manual_seed(42)
    variant = PerturbedCrossDotHead(**config).eval()
    assert (variant.contact_projection.weight - variant.key_projection.weight).abs().max() > 0
    assert (variant(entry, pairs) - baseline(entry, pairs)).abs().max() < .2
    torch.testing.assert_close(variant(entry, pairs), variant(entry, pairs.flip(-1)), atol=0, rtol=0)
    F.cross_entropy(variant(entry, pairs), torch.arange(len(pairs)) % 10).backward()
    assert (variant.contact_projection.weight.grad - variant.key_projection.weight.grad).abs().max() > 1e-8
    reloaded = PerturbedCrossDotHead(**variant.model_config()).eval()
    reloaded.load_state_dict(variant.state_dict())
    torch.testing.assert_close(variant(entry, pairs), reloaded(entry, pairs), atol=0, rtol=0)


@pytest.mark.parametrize('model_class,gradient_name', [
    (SymmetricCrossDotHead, 'key_projection.weight'),
    (PairAdjustedDotHead, 'pair_bias.2.weight'),
])
def test_new_scores_start_as_sphere_and_learn_symmetric_pairs(model_class, gradient_name):
    entry = example()
    pairs = eligible_pairs(entry['mask'])
    config = dict(feature_dim=8, hidden=16, context_layers=2, spherical_dim=64)
    torch.manual_seed(42)
    baseline = PairwiseDistanceHead(**config).eval()
    torch.manual_seed(42)
    variant = model_class(**config).eval()
    torch.testing.assert_close(variant(entry, pairs), baseline(entry, pairs), atol=0, rtol=0)
    labels = torch.arange(len(pairs)) % 10
    F.cross_entropy(variant(entry, pairs), labels).backward()
    grad = dict(variant.named_parameters())[gradient_name].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    with torch.no_grad():
        dict(variant.named_parameters())[gradient_name].add_(grad * .01)
    torch.testing.assert_close(variant(entry, pairs), variant(entry, pairs.flip(-1)), atol=0, rtol=0)
    torch.testing.assert_close(variant(entry, pairs), variant(entry, pairs, chunk_size=7), atol=1e-6, rtol=1e-6)
    restored = model_class(**variant.model_config()).eval()
    restored.load_state_dict(variant.state_dict())
    torch.testing.assert_close(variant(entry, pairs), restored(entry, pairs), atol=0, rtol=0)
