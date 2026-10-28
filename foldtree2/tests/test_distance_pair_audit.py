import torch

from scripts.audit_distance_pair_errors import describe, restraint_preferences


def test_pair_audit_preserves_exact_bin_and_capped_distance_semantics():
    edges = torch.tensor([4., 6., 8., 10., 12., 16., 20., 24., 32.])
    p = torch.zeros(2, 10)
    p[0, 1] = p[1, -1] = 1.
    distances = torch.tensor([5., 100.])
    labels = torch.tensor([1, 9])
    result = describe(p, distances, labels, edges)
    assert result['capped_mae'] == 0.
    assert result['contact_probability'] == result['contact_frequency'] == .5
    assert result['nll'] == 0.
    preferences = restraint_preferences(p, distances, labels, edges)
    assert preferences[0]['preferred_distance_grid_mean'] == 5.
    assert preferences[1]['preferred_at_least_32_fraction'] == 1.


def test_empty_group_is_explicit():
    assert describe(torch.empty(0, 10), torch.empty(0), torch.empty(0, dtype=torch.long),
                    torch.tensor([8.])) == {'pairs': 0}
