import pytest
from scripts.recover_se3_loss_trial import recovery_cost


def test_failed_allocation_and_retry_fit_original_reservation():
    assert recovery_cost(152)==pytest.approx(5.671111111111111)
    assert recovery_cost(152)<6
    with pytest.raises(ValueError,match='original'):recovery_cost(301)
    with pytest.raises(ValueError):recovery_cost(-1)
