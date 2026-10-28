"""Break query/key symmetry for the symmetric cross-dot contact head."""
import torch

from foldtree2.src.distance.contact_dot_experiments import SymmetricCrossDotHead


class PerturbedCrossDotHead(SymmetricCrossDotHead):
    """Start near the shared sphere, with distinct learnable query/key projections."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        generator = torch.Generator(device='cpu').manual_seed(9281)
        with torch.no_grad():
            noise = torch.randn(self.key_projection.weight.shape, generator=generator,
                                dtype=self.key_projection.weight.dtype)
            self.key_projection.weight.add_(noise.to(self.key_projection.weight.device) * .001)
