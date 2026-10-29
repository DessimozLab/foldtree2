"""Local high-precision sums of float32 per-structure gradients."""
import torch
import torch.distributed as dist


class GradientAccumulator:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        if not self.parameters or any(p.dtype != torch.float32 for p in self.parameters):
            raise ValueError('Gradient accumulation expects float32 model parameters')
        device = self.parameters[0].device
        self.buffer = torch.zeros(sum(p.numel() for p in self.parameters), device=device, dtype=torch.float64)
        self.used = torch.zeros(len(self.parameters), device=device, dtype=torch.int64)
        self.count = 0

    @torch.no_grad()
    def collect(self):
        offset = 0
        for i, p in enumerate(self.parameters):
            size = p.numel()
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise ValueError('Sparse gradients are not supported')
                self.buffer[offset:offset+size].add_(p.grad.detach().reshape(-1))
                self.used[i] = 1
            p.grad = None
            offset += size
        self.count += 1

    @torch.no_grad()
    def finish(self, global_batch):
        count = torch.tensor(self.count, device=self.buffer.device, dtype=torch.int64)
        if dist.is_initialized():
            dist.all_reduce(count)
        if int(count) != global_batch:
            raise ValueError('Consumed microbatch count differs from the global batch')
        if dist.is_initialized():
            dist.all_reduce(self.buffer)
            dist.all_reduce(self.used, op=dist.ReduceOp.MAX)
        reduced = (self.buffer / global_batch).float()
        usage = self.used.tolist()
        offset = 0
        for i, p in enumerate(self.parameters):
            size = p.numel()
            p.grad = reduced[offset:offset+size].view_as(p) if usage[i] else None
            offset += size
        self.buffer.zero_(); self.used.zero_(); self.count = 0
