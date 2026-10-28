import pytest
import torch
from scripts.se3.trace_se3_accumulation import reduce_gradients


def test_high_precision_accumulation_preserves_global_mean_across_topologies():
    torch.manual_seed(42)
    grads = list(torch.randn(16, 512))
    reference = torch.stack(grads).double().mean(0)
    for world in [1,2,4,8,16]:
        torch.testing.assert_close(reduce_gradients(grads,world,torch.float64),
                                   reference,atol=0.,rtol=0.)
    four = reduce_gradients(grads,4)
    eight = reduce_gradients(grads,8)
    assert (four != eight).any()
    torch.testing.assert_close(four.double(),reference,atol=2e-7,rtol=1e-6)


def test_invalid_gradient_partition_is_rejected():
    for count,world in [(0,4),(16,3),(16,0)]:
        with pytest.raises(ValueError,match='world size'):
            reduce_gradients([torch.ones(4)]*count,world)


def test_accumulator_preserves_unused_parameters_and_true_global_mean():
    from foldtree2.se3_gradient_accumulation import GradientAccumulator
    active = torch.nn.Parameter(torch.ones(4))
    sometimes = torch.nn.Parameter(torch.ones(4))
    unused = torch.nn.Parameter(torch.ones(4))
    accumulator = GradientAccumulator([active,sometimes,unused])
    torch.manual_seed(42)
    values = torch.randn(16,4)
    for i,value in enumerate(values):
        active.grad = value.clone()
        sometimes.grad = value.clone() if i == 3 else None
        accumulator.collect()
        assert active.grad is None
    accumulator.finish(16)
    torch.testing.assert_close(active.grad,values.double().mean(0).float(),atol=0.,rtol=0.)
    torch.testing.assert_close(sometimes.grad,values[3]/16,atol=0.,rtol=0.)
    assert unused.grad is None
    assert accumulator.count == 0 and not accumulator.buffer.any()


def test_accumulator_rejects_missing_microbatches():
    from foldtree2.se3_gradient_accumulation import GradientAccumulator
    accumulator = GradientAccumulator([torch.nn.Parameter(torch.ones(4))])
    with pytest.raises(ValueError,match='microbatch count'):
        accumulator.finish(16)
