import torch
import torch.nn as nn
from se3_transformer_pytorch import SE3Transformer
from src.utils import quaternion_to_rotation_matrix

class SE3InvariantTransformer(nn.Module):
    """
    SE(3)-invariant Transformer (lucidrains implementation) that predicts per-node rotations,
    translations, and Euler angles when 3D coordinates are available.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
        num_degrees: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Instantiate lucidrains SE3Transformer
        model = SE3Transformer(
            dim = 32,
            heads = 8,
            depth = 1,
            dim_head = 64,
            num_degrees = 2,
            valid_radius = 10,
            attend_sparse_neighbors = True,  # this must be set to true, in which case it will assert that you pass in the adjacency matrix
            num_neighbors = 0,               # if you set this to 0, it will only consider the connected neighbors as defined by the adjacency matrix. but if you set a value greater than 0, it will continue to fetch the closest points up to this many, excluding the ones already specified by the adjacency matrix
        )
        # Final linear to map hidden features to geometry outputs (4 quat + 3 trans + 3 angles)
        self.output_layer = nn.Linear(hidden_dim, 10)

    def forward(self, data):
        """
        Args:
            data: PyG-style graph with:
                x           Tensor[N, node_feature_dim]  # node features
                edge_index  LongTensor[2, E]             # connectivity
                edge_attr   optional edge scalars/vectors
                positions   optional Tensor[N, 3]         # node coords

        Returns:
            R       Tensor[N, 3, 3]  # rotation matrices
            t       Tensor[N, 3]     # translations
            angles  Tensor[N, 3]     # Euler angles
        """
        src, dst = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        unique_batches = torch.unique(data['res'].batch)
        rs = []
        ts = []
        angles_list = []
        for b in unique_batches:
            idx = (data['res'].batch == b).nonzero(as_tuple=True)[0]
            i = torch.arange(len(idx), device=idx.device)
            adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

            if idx.numel() > 2:                
                # SE(3)-invariant encoding
                x_trans = self.transformer(
                    node_feats= torch.cat( [ data['res'].x[idx] ,data['positions'].x[idx] ]),
                    coors = ,
                    adj_mat = adj_mat,
                )

                (feats, coors, mask, adj_mat = adj_mat)
                geom = self.output_layer(x_trans)
                quat = geom[:, :4]
                t = geom[:, 4:7]
                anglesi = geom[:, 7:]
                ri = quaternion_to_rotation_matrix(quat)

                ri = ri.view(-1, 3 , 3)
                ti = ti.view(-1, 3)
                rs.append(ri)
                ts.append(ti)
                angles_list.append(anglesi)
            else:
                rs.append(r[idx])
                ts.append(t[idx])			
                angles_list.append(angles[idx])
        # Concatenate results from all batches
        R = torch.cat(rs, dim=0)
        t = torch.cat(ts, dim=0)
        angles = torch.cat(angles_list, dim=0)
        return R, t, angles

class CoordinateFreeTransformer(nn.Module):
    """
    Standard Transformer that predicts per-node rotations, translations, and Euler angles
    from node features without using coordinate or edge attributes.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Project input features to model dimension
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final linear to map hidden to geometry outputs
        self.output_layer = nn.Linear(hidden_dim, 10)

    def forward(self, data):
        """
        Args:
            data: any object with attribute:
                x  Tensor[N, node_feature_dim]  # node features

        Returns:
            R       Tensor[N, 3, 3]  # rotation matrices
            t       Tensor[N, 3]     # translations
            angles  Tensor[N, 3]     # Euler angles
        """
        x = data.x  # [N, F]

        
        # Project to model dim
        h = self.input_proj(x)  # [N, hidden]
        # Prepare sequence for transformer: [seq_len, batch=1, hidden]
        seq = h.unsqueeze(1)
        # Encode
        enc = self.transformer(seq)  # [N, 1, hidden]
        enc = enc.squeeze(1)        # [N, hidden]
        # Predict geometry
        geom = self.output_layer(enc)
        quat = geom[:, :4]
        t = geom[:, 4:7]
        angles = geom[:, 7:]
        R = quaternion_to_rotation_matrix(quat)
        return R, t, angles
