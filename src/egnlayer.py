
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from e3nn import o3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, MessagePassing

class EGNNLayer(MessagePassing):
    def __init__(self, in_dim, edge_dim, hidden_dim , node_mlp=None, coord_mlp=None):
        super(EGNNLayer, self).__init__(aggr='mean')  # Mean aggregation
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        if node_mlp == None:
            self.node_mlp = nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.node_mlp = node_mlp

        if coord_mlp == None:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),  # Scalar weight for coordinate updates
                nn.Sigmoid()
            )

        else:
            self.coord_mlp = coord_mlp

    def forward(self, x, edge_index , pos , edge_attr = None):
        # Compute messages
        row, col = edge_index
        if edge_attr:
            edge_features = torch.cat([x[row], x[col], edge_attr], dim=-1)
            edge_messages = self.edge_mlp(edge_features)
            # Update node features
            agg_messages = self.propagate(edge_index, x=edge_messages, size=None)
            x = self.node_mlp(torch.cat([x, agg_messages], dim=-1))
        else:
            edge_messages = self.edge_mlp(torch.cat([x[row], x[col]], dim=-1))
            agg_messages = self.propagate(edge_index, x=edge_messages, size=None)
            x = self.node_mlp(torch.cat([x, agg_messages], dim=-1))

        # Equivariant coordinate update
        coord_diff = pos[col] - pos[row]
        coord_updates = self.coord_mlp(edge_messages) * coord_diff
        pos = pos + self.propagate(edge_index, x=coord_updates, size=None)
        return x, pos

    def message(self, x_j):
        return x_j  # Pass messages as they are

    def update(self, aggr_out):
        return aggr_out  # Identity update function


class EGNNLayer_nodes(MessagePassing):
    def __init__(self, in_dim, hidden_dim , node_mlp=None , coord_mlp=None , edge_mlp=None , aggr = 'mean'):
        super(EGNNLayer_nodes, self).__init__(aggr=aggr)  # Mean aggregation

        if edge_mlp == None:
            self.edge_mlp = nn.Sequential(
                nn.Linear(in_dim * 2 + 1, hidden_dim),  # Only use distance as edge feature
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
                )
        else:
            self.edge_mlp = edge_mlp
        if node_mlp == None:
            self.node_mlp = nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
                )
        else:
            self.node_mlp = node_mlp
        if coord_mlp == None:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.coord_mlp = coord_mlp

    def forward(self, x, edge_index, pos):
        row, col = edge_index
        d_ij = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)  # Compute Euclidean distance
        edge_features = torch.cat([x[row], x[col], d_ij], dim=-1)  # Use only distances
        edge_messages = self.edge_mlp(edge_features)

        agg_messages = self.propagate(edge_index, x=edge_messages, size=None)
        x = self.node_mlp(torch.cat([x, agg_messages], dim=-1))

        coord_diff = pos[col] - pos[row]
        coord_updates = self.coord_mlp(edge_messages) * coord_diff
        pos = pos + self.propagate(edge_index, x=coord_updates, size=None)

        return x, pos


class SENEGNNLayer(MessagePassing):
    """ SE(N)-Equivariant Graph Neural Network Layer for N-Dimensional Coordinates """
    def __init__(self, in_dim, hidden_dim, coord_dim):
        """
        Args:
            in_dim: Feature size of input nodes
            hidden_dim: Hidden layer size for message passing
            coord_dim: Dimensionality of coordinate space (N for SE(N))
        """
        super().__init__(aggr="add")
        self.coord_dim = coord_dim  # N-dimensional coordinate space
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.coord_update = nn.Linear(hidden_dim, 1)  # Predict coordinate shifts
        self.rotation_update = nn.Linear(hidden_dim, coord_dim * coord_dim)  # Predict rotation matrix (N×N)
        self.translation_update = nn.Linear(hidden_dim, coord_dim)  # Predict translation vector

    def forward(self, x, edge_index, pos, R=None, t=None):
        """
        Args:
            x: Node features (B, N, in_dim)
            edge_index: Graph connectivity (2, E)
            pos: Node coordinates (B, N, coord_dim) where coord_dim=N
            R: Rotation matrices (B, N, coord_dim, coord_dim) (default: Identity)
            t: Translation vectors (B, N, coord_dim) (default: Zero)
        Returns:
            Updated node features, coordinates, rotations, and translations.
        """
        B, N, _ = pos.shape  # Batch size and number of nodes

        # Initialize R and t if None
        if R is None:
            R = torch.eye(self.coord_dim).expand(B, N, self.coord_dim, self.coord_dim).to(pos.device)  # Identity rotations
        if t is None:
            t = torch.zeros_like(pos).to(pos.device)  # Zero translations

        # Compute new node embeddings
        h = self.linear(x)
        out = self.propagate(edge_index, x=h, pos=pos)

        # Coordinate updates
        coord_update = self.coord_update(out)

        # Predict rotation and translation updates
        R_update = self.rotation_update(out).reshape(B, N, self.coord_dim, self.coord_dim)  # Unnormalized rotation
        t_update = self.translation_update(out)  # Translation update

        # Ensure valid SO(N) rotation matrix using QR decomposition
        Q, _ = torch.linalg.qr(R_update)  # QR decomposition to get an orthogonal matrix
        R_new = torch.matmul(Q, R)  # Apply learned rotation update

        # Apply translation update
        t_new = t + t_update

        return out, pos + coord_update, R_new, t_new  # Updated representations




class StackedSENEGNN(nn.Module):
    """ SE(N)-EGNN with Multiple Layers and Optional R, t Handling """
    def __init__(self, in_dim, hidden_dim, coord_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            SENEGNNLayer(in_dim if i == 0 else hidden_dim, hidden_dim, coord_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Final projection

    def forward(self, x, edge_index, pos, R=None, t=None):
        """
        Args:
            x: Node features (B, N, in_dim)
            edge_index: Graph connectivity (2, E)
            pos: Node coordinates (B, N, coord_dim)
            R: Rotation matrices (B, N, coord_dim, coord_dim) (default: Identity)
            t: Translation vectors (B, N, coord_dim) (default: Zero)
        """
        for layer in self.layers:
            x, pos, R, t = layer(x, edge_index, pos, R, t)

        z = self.fc(x)  # Final latent representation
        return z, pos, R, t  # Final outputs

class SENEGNNLayer_heterograph(MessagePassing):
    """ SE(N)-Equivariant Graph Neural Network Layer for Heterogeneous Graphs. """
    def __init__(self, in_dim, hidden_dim, coord_dim, edge_types):
        """
        Args:
            in_dim: Feature size of input nodes.
            hidden_dim: Hidden layer size for message passing.
            coord_dim: Dimensionality of coordinate space (N for SE(N)).
            edge_types: List of edge types in the heterogeneous graph.
        """
        super().__init__(aggr="add")
        self.coord_dim = coord_dim
        self.edge_types = edge_types  # Edge types in the heterogeneous graph
        self.linear = nn.ModuleDict({edge: nn.Linear(in_dim, hidden_dim) for edge in edge_types})
        self.coord_update = nn.ModuleDict({edge: nn.Linear(hidden_dim, 1) for edge in edge_types})
        self.rotation_update = nn.ModuleDict({edge: nn.Linear(hidden_dim, coord_dim * coord_dim) for edge in edge_types})
        self.translation_update = nn.ModuleDict({edge: nn.Linear(hidden_dim, coord_dim) for edge in edge_types})

    def forward(self, x_dict, edge_index_dict, pos_dict, R_dict=None, t_dict=None):
        """
        Args:
            x_dict: Dictionary of node type -> node features (B, N, in_dim).
            edge_index_dict: Dictionary of edge type -> connectivity (2, E).
            pos_dict: Dictionary of node type -> coordinates (B, N, coord_dim).
            R_dict: Dictionary of node type -> rotation matrices (B, N, coord_dim, coord_dim) (default: Identity).
            t_dict: Dictionary of node type -> translation vectors (B, N, coord_dim) (default: Zero).
        Returns:
            Updated node features, coordinates, rotations, and translations.
        """
        # Initialize R and t if None
        if R_dict is None:
            R_dict = {node: torch.eye(self.coord_dim).expand(x.shape[0], x.shape[1], self.coord_dim, self.coord_dim).to(x.device) for node, x in x_dict.items()}
        if t_dict is None:
            t_dict = {node: torch.zeros_like(pos).to(pos.device) for node, pos in pos_dict.items()}

        out_dict, pos_new_dict, R_new_dict, t_new_dict = {}, {}, {}, {}

        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type  # Extract node types from edge
            if src not in x_dict or dst not in x_dict:
                continue  # Skip if either node type is missing

            # Feature update
            h = self.linear[edge_type](x_dict[src])
            out = self.propagate(edge_index, x=h, pos=pos_dict[src])

            # Coordinate updates
            coord_update = self.coord_update[edge_type](out)

            # Predict rotation and translation updates
            R_update = self.rotation_update[edge_type](out).reshape(-1, self.coord_dim, self.coord_dim)
            t_update = self.translation_update[edge_type](out)

            # Ensure valid SO(N) rotation matrix using QR decomposition
            Q, _ = torch.linalg.qr(R_update)  # QR decomposition for valid rotation
            R_new = torch.matmul(Q, R_dict[src])  # Apply learned rotation update

            # Apply translation update
            t_new = t_dict[src] + t_update

            out_dict[src] = out
            pos_new_dict[src] = pos_dict[src] + coord_update
            R_new_dict[src] = R_new
            t_new_dict[src] = t_new

        return out_dict, pos_new_dict, R_new_dict, t_new_dict

class StackedSENEGNN_hetero(nn.Module):
    """ SE(N)-EGNN with Multiple Layers using HeteroConv """
    def __init__(self, in_dim, hidden_dim, coord_dim, num_layers, edge_types):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroConv({edge: SENEGNNLayer(in_dim if i == 0 else hidden_dim, hidden_dim, coord_dim, edge_types)
                        for edge in edge_types}, aggr="sum")  # Aggregate messages
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Final projection

    def forward(self, x_dict, edge_index_dict, pos_dict, R_dict=None, t_dict=None):
        """
        Args:
            x_dict: Dictionary of node type -> node features.
            edge_index_dict: Dictionary of edge type -> connectivity.
            pos_dict: Dictionary of node type -> coordinates.
            R_dict: Dictionary of node type -> rotation matrices (optional).
            t_dict: Dictionary of node type -> translation vectors (optional).
        """
        for layer in self.layers:
            x_dict, pos_dict, R_dict, t_dict = layer(x_dict, edge_index_dict, pos_dict, R_dict, t_dict)

        z_dict = {key: self.fc(x) for key, x in x_dict.items()}  # Final embeddings
        return z_dict, pos_dict, R_dict, t_dict  # Final outputs


class GaussianDiffusion:
    """ Implements the forward (noise addition) and reverse (denoising) process. """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # Cumulative product of alphas

    def forward_process(self, X_0, t, noise=None):
        """ Adds Gaussian noise to coordinates X_0 at timestep t. """
        if noise is None:
            noise = torch.randn_like(X_0)
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1)  # Shape (B, 1, 1)
        return torch.sqrt(alpha_t) * X_0 + torch.sqrt(1 - alpha_t) * noise, noise


class SENEGNNLayer(MessagePassing):
    """ SE(N)-Equivariant Graph Neural Network Layer for Diffusion Model. """
    def __init__(self, in_dim, hidden_dim, coord_dim, edge_types):
        super().__init__(aggr="add")
        self.coord_dim = coord_dim
        self.edge_types = edge_types
        self.linear = nn.ModuleDict({edge: nn.Linear(in_dim, hidden_dim) for edge in edge_types})
        self.coord_update = nn.ModuleDict({edge: nn.Linear(hidden_dim, 1) for edge in edge_types})
        self.rotation_update = nn.ModuleDict({edge: nn.Linear(hidden_dim, coord_dim * coord_dim) for edge in edge_types})
        self.translation_update = nn.ModuleDict({edge: nn.Linear(hidden_dim, coord_dim) for edge in edge_types})

    def forward(self, x_dict, edge_index_dict, pos_dict, R_dict=None, t_dict=None):
        if R_dict is None:
            R_dict = {node: torch.eye(self.coord_dim).expand(x.shape[0], x.shape[1], self.coord_dim, self.coord_dim).to(x.device) for node, x in x_dict.items()}
        if t_dict is None:
            t_dict = {node: torch.zeros_like(pos).to(pos.device) for node, pos in pos_dict.items()}

        out_dict, pos_new_dict, R_new_dict, t_new_dict = {}, {}, {}, {}

        for edge_type, edge_index in edge_index_dict.items():
            src, _, dst = edge_type
            if src not in x_dict or dst not in x_dict:
                continue

            h = self.linear[edge_type](x_dict[src])
            out = self.propagate(edge_index, x=h, pos=pos_dict[src])

            coord_update = self.coord_update[edge_type](out)
            R_update = self.rotation_update[edge_type](out).reshape(-1, self.coord_dim, self.coord_dim)
            t_update = self.translation_update[edge_type](out)

            Q, _ = torch.linalg.qr(R_update)
            R_new = torch.matmul(Q, R_dict[src])
            t_new = t_dict[src] + t_update

            out_dict[src] = out
            pos_new_dict[src] = pos_dict[src] + coord_update
            R_new_dict[src] = R_new
            t_new_dict[src] = t_new

        return out_dict, pos_new_dict, R_new_dict, t_new_dict

class SE_N_Diffusion(nn.Module):
    """ Diffusion Model with SE(N)-EGNN as the denoising network """
    def __init__(self, in_dim, hidden_dim, coord_dim, num_layers, edge_types):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroConv({edge: SENEGNNLayer(in_dim if i == 0 else hidden_dim, hidden_dim, coord_dim, edge_types)
                        for edge in edge_types}, aggr="sum")
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict, pos_dict, R_dict=None, t_dict=None):
        for layer in self.layers:
            x_dict, pos_dict, R_dict, t_dict = layer(x_dict, edge_index_dict, pos_dict, R_dict, t_dict)

        return pos_dict, R_dict, t_dict


def fape_loss_n_dim(X_pred, X_true, R_pred, t_pred, R_true, t_true, mask=None):
    """ Compute FAPE Loss for N-dimensional embeddings. """
    X_pred_transformed = torch.einsum('bndd,bnd->bnd', R_pred, X_pred) + t_pred
    X_true_transformed = torch.einsum('bndd,bnd->bnd', R_true, X_true) + t_true

    loss = F.mse_loss(X_pred_transformed, X_true_transformed, reduction='none')

    if mask is not None:
        loss = loss * mask.unsqueeze(-1)

    return loss.mean()



class SoftFAPELoss(nn.Module):
    """
    Implements Frame Aligned Point Error (FAPE) with Soft Alignments for two point clouds.
    This version replaces hard nearest-neighbor matching with a probability-weighted alignment.
    """
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Controls the sharpness of the soft alignment (lower = harder matching).
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, X_pred, X_true, R_pred, t_pred, R_true, t_true):
        """
        Args:
            X_pred: Predicted point cloud (B, N, D)
            X_true: Ground truth point cloud (B, M, D)
            R_pred: Predicted rotation matrices (B, D, D)
            t_pred: Predicted translation vectors (B, D)
            R_true: Ground truth rotation matrices (B, D, D)
            t_true: Ground truth translation vectors (B, D)
        Returns:
            fape_loss: Soft alignment FAPE loss (scalar)
        """
        B, N, D = X_pred.shape
        M = X_true.shape[1]

        # Apply rigid transformations
        X_pred_transformed = torch.einsum('bij,bnj->bni', R_pred, X_pred) + t_pred.unsqueeze(1)  # (B, N, D)
        X_true_transformed = torch.einsum('bij,bmj->bmi', R_true, X_true) + t_true.unsqueeze(1)  # (B, M, D)

        # Compute pairwise squared Euclidean distances
        dist_sq = torch.cdist(X_pred_transformed, X_true_transformed, p=2).pow(2)  # (B, N, M)

        # Compute soft alignment using Gaussian kernel with learnable temperature
        soft_alignment = F.softmax(-dist_sq / self.temperature, dim=-1)  # (B, N, M)

        # Compute soft FAPE loss
        weighted_distances = (soft_alignment * dist_sq).sum(dim=-1)  # (B, N)
        fape_loss = weighted_distances.mean()  # Scalar loss

        return fape_loss
