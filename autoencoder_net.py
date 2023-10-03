
import biographs as bg





molecule = bg.Pmolecule('Users/rdora/1eei.pdb')
# biopython molecule structural model
mol_model = pmolecule.model
# networkx graph, by default 5 angstrom
network = molecule.network()
list(network.nodes)[:10]
# Output
['G26', 'G27', 'G24', 'G25', 'G22', 'G23', 'G20', 'G21', 'G28', 'G29']

####use the biographs library to create a graph from a pdb file


####use the autoencode net to create embeddings


####discretize the embeddings into 256 characters

#### transform pdbs into ascii




import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define autoencoder model
class Autoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Autoencoder, self).__init__()
        self.encoder = GCNConv(in_channels, hidden_channels)
        self.decoder = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        z = F.relu(self.encoder(x, edge_index))
        x_hat = self.decoder(z, edge_index)
        return z, x_hat

# Instantiate model and optimizer
model = Autoencoder(dataset.num_node_features, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    z, x_hat = model(dataset.x, dataset.edge_index)
    loss = F.mse_loss(x_hat, dataset.x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

# Extract node embeddings
model.eval()
with torch.no_grad():
    z, _ = model(dataset.x, dataset.edge_index)
node_embeddings = z.detach().cpu().numpy()

print('Node embeddings shape:', node_embeddings.shape)



#####


import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_undirected

# Load dataset
dataset = Planetoid(root='data', name='Cora')

# Convert graph to undirected
data = dataset[0]
data = to_undirected(data)

# Define model
class HeteroGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HeteroGCN, self).__init__()
        self.encoder_conv1 = GCNConv(in_channels, hidden_channels)
        self.encoder_conv2 = GATConv(hidden_channels, hidden_channels)
        self.decoder_conv1 = GCNConv(hidden_channels, out_channels)
        self.decoder_conv2 = GATConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        # Encoder
        x = F.relu(self.encoder_conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.encoder_conv2(x, edge_index)
        encoded = x

        # Decoder
        x = F.relu(self.decoder_conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.decoder_conv2(x, edge_index)
        return encoded, x

# Instantiate model and optimizer
model = HeteroGCN(data.num_node_features, 16, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



###
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load dataset
dataset = Planetoid(root='data', name='Cora')

# Define model
class NodeAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeAutoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Instantiate model and optimizer
model = NodeAutoencoder(dataset.num_node_features, 16, dataset.num_node_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(dataset.data.x, dataset.data.edge_index)
    loss = F.mse_loss(out, dataset.data.x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

# Extract node embeddings
model.eval()
with torch.no_grad():
    out = model(dataset.data.x, dataset.data.edge_index)
node_embeddings = out.detach().cpu().numpy()

print('Node embeddings shape:', node_embeddings.shape)


####

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGPooling

# Load dataset
dataset = Planetoid(root='data', name='Cora')

# Define model
class NodeAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeAutoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = SAGPooling(hidden_channels, ratio=0.5)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        return x

# Instantiate model and optimizer
model = NodeAutoencoder(dataset.num_node_features, 16, dataset.num_node_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(dataset.data.x, dataset.data.edge_index, dataset.data.batch)
    loss = F.mse_loss(out, dataset.data.x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

# Extract node embeddings
model.eval()
with torch.no_grad():
    out = model(dataset.data.x, dataset.data.edge_index, dataset.data.batch)
node_embeddings = out.detach().cpu().numpy()

print('Node embeddings shape:', node_embeddings.shape)


