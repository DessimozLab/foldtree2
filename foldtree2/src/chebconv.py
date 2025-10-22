import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import get_laplacian

class StableChebConv(MessagePassing):
    r"""
    Stable-ChebNet layer with antisymmetric weights and optional spectral norm rescaling.

    Update rule:
        X_{t+1} = X_t + h * ( sum_k T_k(L_hat) X_t A_k - gamma * X_t )

    where:
        - h is a learnable positive step size (softplus(raw_h))
        - A_k = M_k - M_k^T (antisymmetric)
        - spectral_norm option rescales A_k so ||A_k||_2 <= spectral_clip

    Args:
        in_channels (int)
        out_channels (int)
        hidden_channels (int, optional): internal square dim (default=in_channels)
        K (int): Chebyshev order
        normalization (str): Laplacian normalization ('sym', 'rw', or None)
        step_size (float): initial value of step size h
        learnable_step (bool): if True, h is a learnable positive parameter
        gamma (float): small damping coefficient (>=0)
        bias (bool): add bias after output projection
        spectral_norm (bool): if True, rescale antisymmetric weights
        spectral_clip (float): maximum spectral norm allowed
    """

    def __init__(self,
                 in_channels,  # Can be int or tuple (-1, -1)
                 out_channels: int = None,
                 hidden_channels: int = None,
                 K: int = 5,
                 normalization: str = 'sym',
                 step_size: float = 0.2,
                 learnable_step: bool = True,
                 gamma: float = 0.0,
                 bias: bool = True,
                 spectral_norm: bool = False,
                 spectral_clip: float = 1.0):
        super().__init__(aggr='add')
        assert K >= 1
        assert normalization in [None, 'sym', 'rw']
        self.K = K
        self.normalization = normalization
        self.h = float(step_size)
        self.learnable_step = learnable_step
        self.gamma = gamma
        self.spectral_norm = spectral_norm
        self.spectral_clip = spectral_clip
        self._bias_enabled = bias

        # Handle in_channels as tuple (-1, -1) or int
        if isinstance(in_channels, tuple):
            assert in_channels == (-1, -1), \
                "Only (-1, -1) tuple is supported for in_channels"
            self.in_channels = -1  # Will be inferred later
        else:
            self.in_channels = in_channels

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Learnable step size (can be initialized immediately)
        if learnable_step:
            inv_sp = torch.log(torch.exp(torch.tensor(step_size)) - 1.0)
            self.raw_h = nn.Parameter(inv_sp.clone().detach())
        else:
            self.register_parameter('raw_h', None)

        # Always try to create parameters if possible
        if self.in_channels > 0:
            self._create_parameters()
        else:
            # Set parameters to None for lazy init
            self.in_lin = None
            self.out_lin = None
            self.M = None
            self.bias = None

    def _create_parameters(self):
        """Create all parameters once in_channels is known."""
        if self.hidden_channels is None:
            self.hidden_channels = self.in_channels
        
        # Input/output projections
        if self.in_channels != self.hidden_channels:
            self.in_lin = Linear(
                self.in_channels, self.hidden_channels,
                bias=False, weight_initializer='glorot')
        else:
            self.in_lin = nn.Identity()
            
        if self.hidden_channels != self.out_channels:
            self.out_lin = Linear(
                self.hidden_channels, self.out_channels,
                bias=False, weight_initializer='glorot')
        else:
            self.out_lin = nn.Identity()

        # Raw weight matrices
        self.M = nn.ParameterList([
            nn.Parameter(torch.empty(
                self.hidden_channels, self.hidden_channels))
            for _ in range(self.K)
        ])

        if self._bias_enabled:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.M is not None:
            for Mk in self.M:
                nn.init.xavier_uniform_(Mk)
        if hasattr(self, 'in_lin') and isinstance(self.in_lin, Linear):
            self.in_lin.reset_parameters()
        if hasattr(self, 'out_lin') and isinstance(self.out_lin, Linear):
            self.out_lin.reset_parameters()
        if hasattr(self, 'bias') and self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def _norm(self, edge_index, num_nodes, edge_weight=None,
              lambda_max=None, dtype=None, batch=None):
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, self.normalization, dtype, num_nodes)
        assert edge_weight is not None
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(
                lambda_max, dtype=dtype, device=edge_index.device)
        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1
        return edge_index, edge_weight

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def _cheb_apply(self, x: Tensor, edge_index: Tensor, norm: Tensor):
        Tx_0 = x
        outs = [Tx_0]
        if self.K >= 2:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            outs.append(Tx_1)
            for _ in range(2, self.K):
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
                Tx_2 = 2. * Tx_2 - Tx_0
                outs.append(Tx_2)
                Tx_0, Tx_1 = Tx_1, Tx_2
        return outs

    def _step_size(self) -> Tensor:
        if self.learnable_step:
            return F.softplus(self.raw_h) + 1e-8
        else:
            return torch.tensor(self.h, device=self.M[0].device)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Tensor = None,
                batch: Tensor = None,
                lambda_max: Tensor = None) -> Tensor:
        
        # Lazy initialization if needed
        if self.in_channels == -1:
            self.in_channels = x.size(-1)
            if self.out_channels is None:
                self.out_channels = self.in_channels
            self._create_parameters()
            #make sure all parameters are on the same device as x
            self.to(x.device)
        
        x_h = self.in_lin(x)
        edge_index, norm = self._norm(edge_index, x_h.size(0), edge_weight,
                                      lambda_max, dtype=x_h.dtype, batch=batch)
        Txs = self._cheb_apply(x_h, edge_index, norm)

        F_accum = 0.0
        for k in range(self.K):
            Mk = self.M[k]
            A_k = Mk - Mk.T
            if self.spectral_norm:
                with torch.no_grad():
                    norm_val = torch.linalg.matrix_norm(A_k, ord=2)
                if norm_val > 0:
                    A_k = (self.spectral_clip / norm_val) * A_k
            F_accum = F_accum + (Txs[k] @ A_k)

        h = self._step_size()
        x_h_next = x_h + h * (F_accum - self.gamma * x_h)

        out = self.out_lin(x_h_next)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        in_ch = self.in_channels if self.in_channels != -1 else "(-1, -1)"
        return (f'{self.__class__.__name__}(in={in_ch}, '
                f'hid={self.hidden_channels}, out={self.out_channels}, '
                f'K={self.K}, learnable_h={self.learnable_step}, '
                f'spectral_norm={self.spectral_norm}, '
                f'gamma={self.gamma}, normalization={self.normalization})')
