import torch
from typing import Tuple

class Lion(torch.optim.Optimizer):
    """
    Lion optimizer (EvoLved Sign Momentum) - Google Brain 2023.
    
    Uses only the sign of momentum for updates, resulting in:
    - 50% less memory than Adam (only 1 state per parameter)
    - Simpler computation (no sqrt or division)
    - Often faster convergence
    
    Note: Requires ~3-10x lower learning rate than AdamW.
    Recommended: lr=1e-5 to 3e-5, weight_decay=0.1 to 0.3
    """
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), 
                 weight_decay: float = 0.0):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate (use 3-10x lower than AdamW)
            betas: Coefficients for computing running average (beta1, beta2)
                   beta1: for update interpolation (default: 0.9)
                   beta2: for momentum decay (default: 0.99)
            weight_decay: Decoupled weight decay (default: 0.0)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize momentum state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Decoupled weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Lion update: use sign of interpolated momentum
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update momentum for next iteration
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss

