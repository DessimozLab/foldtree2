import pytest
import torch
from scripts.se3.se3_extraction_parity import compare_entries


def test_runtime_drift_is_explicit_and_repeatability_remains_strict():
    a = {'features': torch.tensor([1.]), 'tokens': torch.tensor([1]), 'mask': torch.tensor([True])}
    b = dict(a, features=torch.tensor([1.000244]))
    with pytest.raises(AssertionError):
        compare_entries(a, b)
    report = compare_entries(a, b, True)
    assert 2e-4 < report['features']['max_absolute'] < 3e-4
    compare_entries(b, b)


@pytest.mark.parametrize('key', ['tokens', 'target', 'mask', 'codebook'])
def test_runtime_permission_does_not_allow_changed_exact_fields(key):
    a = {key: torch.zeros(2)}
    b = {key: torch.ones(2)*1e-5}
    with pytest.raises(AssertionError):
        compare_entries(a, b, True)


def test_runtime_drift_remains_bounded():
    with pytest.raises(AssertionError):
        compare_entries({'features': torch.zeros(2)}, {'features': torch.ones(2)*.01}, True)
    with pytest.raises(AssertionError):
        compare_entries({'features': torch.zeros(2)}, {'features': torch.full((2,), float('nan'))}, True)


def test_converted_angles_allow_one_ulp_but_not_material_label_changes():
    a = torch.ones(2)
    b = torch.nextafter(a, torch.full_like(a, float('inf')))
    compare_entries({'angles': a}, {'angles': b})
    with pytest.raises(AssertionError):
        compare_entries({'angles': a}, {'angles': a + .001}, True)


def test_contact_topology_drift_is_reported_and_bounded():
    a = torch.ones(20, 20, dtype=torch.bool)
    b = a.clone(); b[0, 1] = b[1, 0] = False
    report = compare_entries({'dot': a}, {'dot': b}, True)
    assert report['dot']['changed_fraction'] == .005
    b[0] = False
    with pytest.raises(ValueError):
        compare_entries({'dot': a}, {'dot': b}, True)
