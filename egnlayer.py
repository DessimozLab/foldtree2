import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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
