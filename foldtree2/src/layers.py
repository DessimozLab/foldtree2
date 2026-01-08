import torch
import torch.nn as nn

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Learnable_Positional_Encoding(nn.Module):
	def __init__(self, dim, max_len=1024):
		super(Learnable_Positional_Encoding, self).__init__()
		self.pos_embedding = nn.Embedding(max_len, dim)

	def forward(self, x):
		# x shape: (batch_size, seq_len, dim)
		seq_len = x.size(1)
		position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
		position_ids = position_ids.unsqueeze(0).expand_as(x[:, :, 0])  # (batch_size, seq_len)
		pos_embeddings = self.pos_embedding(position_ids)
		return x + pos_embeddings

class Position_MLP(torch.nn.Module):
	def __init__(self, in_channels=256, hidden_channels=[128, 64], out_channels=32, dropout=0.1):
		super(Position_MLP, self).__init__()
		layers = []
		layers.append( nn.Linear(in_channels, hidden_channels[0]) )
		layers.append( nn.GELU() )
		layers.append( nn.Dropout(dropout) )
		for i in range(1, len(hidden_channels)):
			layers.append( nn.Linear(hidden_channels[i-1], hidden_channels[i]) )
			layers.append( nn.GELU() )
			layers.append( nn.Dropout(dropout) )
		layers.append( nn.Linear(hidden_channels[-1], out_channels) )
		self.mlp = nn.Sequential( *layers )

	def forward(self, x):
		return self.mlp(x)