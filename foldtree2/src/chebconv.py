import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian
        
from torch_geometric.typing import OptPairTensor
from torch import Tensor
from typing import Union

class StableChebConv(MessagePassing):
    r"""
    Stable-ChebNet Convolution Layer implementing Chebyshev polynomial filtering with 
    antisymmetric weight parametrization and forward-Euler update.
    
    Args:
        in_channels (int): Number of input features (size of each input node feature vector). 
                           Can be set to `-1` for lazy initialization.
        out_channels (int): Number of output features (size of each output node feature vector). 
                            Can be set to `-1` for lazy initialization (in which case it defaults to `in_channels` on first use).
        K (int): Chebyshev polynomial order (number of hops included in the filter). Must be >= 1.
        normalization (str, optional): Laplacian normalization scheme; one of `None`, `"sym"`, or `"rw"`. 
                                       Default is `"sym"` (symmetric normalization, i.e. $L = I - D^{-1/2} A D^{-1/2}$).
        spectral_norm (bool, optional): If True, apply spectral normalization to each antisymmetric weight matrix (default: False).
        bias (bool, optional): If False, no bias term is added. Default: True.
        step_size (float, optional): Initial value for the learnable step size $\eta$. Default: 0.1.
    """
    def __init__(self, in_channels: int , out_channels: int, K: int,
                 normalization: str = 'sym', spectral_norm: bool = False,
                 bias: bool = True, step_size: float = 0.1, explain = False , **kwargs):
        #super(MessagePassing, self).__init__( **kwargs)#aggr='add', **kwargs)
        super(StableChebConv, self).__init__()

        assert K > 0, "K (Chebyshev polynomial order) must be at least 1."
        assert normalization in [None, 'sym', 'rw'], "Invalid normalization. Use None, 'sym', or 'rw'."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.spectral_norm = spectral_norm
        self.explain = explain
        self.needs_init = (in_channels == -1 or out_channels == -1)
        # Lazy initialization placeholders
        self.weight = None  # will be Parameter or UninitializedParameter
        self.bias = None
        self.lin_skip = None  # for projecting skip connection if needed
        # Learnable step size parameter (scalar), ensure positivity via softplus in forward:
        self._step_param = Parameter(torch.tensor([step_size], dtype=torch.float))  

        # If dimensions are known (non-lazy), initialize parameters now:
        if in_channels != -1 and out_channels != -1:
            self._init_weights(in_channels, out_channels, bias)
        else:
            # Register as uninitialized parameters for lazy init:
            if bias:
                # Use UninitializedParameter if available, else placeholder
                try:
                    from torch.nn.parameter import UninitializedParameter
                    self.bias = UninitializedParameter()
                except ImportError:
                    # Fallback: register as None (will initialize later)
                    self.register_parameter('bias', None)
            else:
                self.register_parameter('bias', None)
            # We won't create self.weight or self.lin_skip until forward when shapes are known.

    def _init_weights(self, in_channels: int, out_channels: int, bias: bool):
        """Initialize weight and bias parameters given known in/out channels."""
        # Determine if antisymmetric enforcement is applicable (only if in==out):
        self._antisym = (in_channels == out_channels)
        # Initialize weight parameter:
        if self._antisym:
            # Use a full [K, out, out] matrix, will enforce antisymmetry in forward
            self.weight = Parameter(torch.Tensor(self.K, out_channels, out_channels))
        else:
            # Use standard [K, in, out] weight matrix for each Cheb term
            if type(in_channels) is tuple:
                #change to tuple of ints
                print( in_channels)
                in_channels = ( int(in_channels[0]) , int(in_channels[1]) )
            self.weight = Parameter(torch.Tensor(self.K, in_channels, out_channels))
        # Initialize skip connection linear if needed (for dimension mismatch):
        if in_channels != out_channels:
            self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)
        # Initialize bias parameter:
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        # Reset parameters (initialize values)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters using appropriate initializations."""
        # Only initialize if weight is allocated:
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)  # glorot initialization for weight
        if self.lin_skip is not None:
            # For skip linear, we can use identity init if shapes match partially, 
            # but if using skip only for dimension change, use Xavier:
            nn.init.xavier_uniform_(self.lin_skip.weight)
        if self.bias is not None and isinstance(self.bias, Parameter):
            nn.init.zeros_(self.bias)
        # Initialize step size parameter (small positive):
        self._step_param.data.fill_(0.1)  # or retain existing data as set in __init__

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_weight: Tensor = None,
        batch: Tensor = None,
        lambda_max: Tensor = None,
    ) -> Tensor:
        """Apply the stable Chebyshev convolution on input features `x`."""

        # Handle OptPairTensor: (x_src, x_dst) or plain Tensor
        if isinstance(x, Tensor):
            x_src, x_dst = x, None
        else:
            x_src, x_dst = x
        x_target = x_dst if x_dst is not None else x_src

        # Lazy initialization: infer in/out channels from input on first run
        needs_init = (
            self.in_channels == -1
            or self.out_channels == -1
            or self.weight is None
        )
        if needs_init:
            print("Lazy initialization of StableChebConv parameters.")
            print(x_src)
            Fin = x_src.size(1)
            if self.in_channels == -1:
                self.in_channels = Fin
            if self.out_channels == -1:
                self.out_channels = self.in_channels  # default to same dim
            # Now initialize weights with determined dimensions
            self._init_weights(
                self.in_channels,
                self.out_channels,
                bias=(self.bias is not None),
            )
            # Move newly created parameters to same device as input
            self.to(x_src.device)

        # Validate lambda_max if needed:
        if self.normalization != "sym":
            if lambda_max is None:
                raise ValueError(
                    "`lambda_max` must be provided for normalization = {}.".format(
                        self.normalization
                    )
                )
        # Default lambda_max for symmetric normalization:
        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x_src.dtype, device=x_src.device)

        # Compute normalized/scaled Laplacian edge weights (norm) for propagation:
        edge_index, norm = self._compute_norm(
            edge_index,
            x_src.size(0),
            edge_weight,
            self.normalization,
            lambda_max=lambda_max,
            dtype=x_src.dtype,
            batch=batch,
        )

        # Chebyshev polynomial propagation:
        Tx_0 = x_src  # T0 * x

        # Prepare weight matrices (enforce antisymmetry if applicable):
        if hasattr(self, "_antisym") and self._antisym:
            W = self.weight
            W_asym = W - W.transpose(1, 2)
            if self.spectral_norm:
                W_list = []
                for k in range(self.K):
                    Wk = W_asym[k]
                    try:
                        sigma_max = torch.linalg.svdvals(Wk)[0]
                    except RuntimeError:
                        sigma_max = torch.svd(Wk, compute_uv=False)[1][0]
                    if sigma_max > 0:
                        Wk = Wk / sigma_max
                    W_list.append(Wk)
                W_effective = torch.stack(W_list, dim=0)
            else:
                W_effective = 0.5 * W_asym
        else:
            W_effective = self.weight

        # Ensure weights are on same device as features
        W_effective = W_effective.to(x_src.device)

        # Compute convolution output using Chebyshev basis:
        out = Tx_0.matmul(W_effective[0])
        if self.K > 1:
            Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)
            out += Tx_1.matmul(W_effective[1])
            for k in range(2, self.K):
                Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
                out += Tx_2.matmul(W_effective[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        # If we had separate target features, restrict output to target nodes
        if x_dst is not None:
            out = out[: x_dst.size(0)]

        # Add bias if present:
        if self.bias is not None:
            out = out + self.bias.to(out.device)

        # Compute positive step size and combine with skip connection:
        step = F.softplus(self._step_param)  # positive scalar
        
        if self.lin_skip is not None:
        
            print(x_target)
            print(x_target.shape)
            
            out = self.lin_skip(x_target) + step * out
        else:
            out = x_target + step * out
        return out


    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Message computation: multiply neighbor feature x_j by norm (Laplacian weight)
        return norm.view(-1, 1) * x_j

    @staticmethod
    def _compute_norm(edge_index: torch.Tensor, num_nodes: int,
                      edge_weight: torch.Tensor, normalization: str,
                      lambda_max: torch.Tensor, dtype: torch.dtype, batch: torch.Tensor):
        """
        Compute scaled Laplacian edge weights for Chebyshev propagation.
        Returns (edge_index, norm_weights) where norm_weights correspond to 
        (2/lambda_max * L - I) entries on each edge.
        """
        # Remove self-loops to avoid double counting:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        # Compute graph Laplacian (unscaled) for given normalization:
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype=dtype, num_nodes=num_nodes)
        # If batch-wise lambda_max is provided as a tensor, match each edge to its graph's lambda_max:
        if batch is not None and isinstance(lambda_max, torch.Tensor) and lambda_max.dim() > 0:
            # Use source node of edge (edge_index[0]) to index into lambda_max per graph
            lambda_max_edge = lambda_max[batch[edge_index[0]].to(lambda_max.device)]
        else:
            lambda_max_edge = lambda_max
        # Scale Laplacian weights by 2/lambda_max:
        edge_weight = edge_weight * (2.0 / lambda_max_edge)
        # Replace any infinities (from zero lambda_max) with 0:
        edge_weight[torch.isinf(edge_weight)] = 0.0
        # Subtract identity: add self-loops with weight -1 to achieve (2L/λ_max - I)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1., num_nodes=num_nodes)
        return edge_index, edge_weight
