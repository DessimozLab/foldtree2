import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
import foldtree2_ecddcd as ft2
import pickle
import pandas as pd

# Load data
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

# Initialize dataset and DataLoader
datadir = '../../datasets/'
struct_dat = ft2.StructureDataset('structs_training_godnodemk3.h5')
BATCH_SIZE = 60


converter = ft2.PDB2PyG()

filename = './1eei.pdb'

data_sample =converter.struct2pyg( filename, verbose=False)
print(data_sample)

ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]


train_loader = DataLoader(struct_dat, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

devices = torch.cuda.device_count()
print(f"Using {devices} GPUs")

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)

class GNNModel(pl.LightningModule):
    def __init__(self, ndim, ndim_godnode, encoder_layers=3, decoder_layers=5, learning_rate=1e-3):
        super(GNNModel, self).__init__()
        
        self.encoder = ft2.HeteroGAE_Encoder(
            in_channels={'res': ndim, 'godnode': ndim_godnode},
            hidden_channels={
                ('res', 'backbone', 'res'): [100] * encoder_layers,
                ('res', 'contactPoints', 'res'): [100] * encoder_layers
            },
            layers=encoder_layers,
            out_channels=10,
            metadata=converter.metadata,
            num_embeddings=40,
            commitment_cost=0.8,
            encoder_hidden=100,
            nheads=1,
            average=False,
            reset_codes=False,
            dropout_p=0.001,
            separated=True,
            flavor='sage'
        )
        
        self.decoder = ft2.HeteroGAE_Decoder(
            in_channels={'res': self.encoder.out_channels, 'godnode4decoder': ndim_godnode, 'foldx': 23},
            hidden_channels={
                ('res', 'backbone', 'res'): [300] * decoder_layers,
                ('res', 'informs', 'godnode4decoder'): [300] * decoder_layers,
                ('godnode4decoder', 'informs', 'res'): [300] * decoder_layers
            },
            layers=decoder_layers,
            metadata=converter.metadata,
            amino_mapper=converter.aaindex,
            flavor='sage',
            Xdecoder_hidden=100,
            PINNdecoder_hidden=[100, 50, 10],
            contactdecoder_hidden=[100, 50, 10],
            nheads=1,
            dropout=0.001,
            AAdecoder_hidden=[100, 50, 20]
        )

        self.batchnorm_fx = torch.nn.BatchNorm1d(23)
        self.learning_rate = learning_rate
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        # Load mean and variance for normalization
        mean = pd.read_csv('foldxmean.csv', index_col=0).values
        variance = pd.read_csv('foldxvariance.csv', index_col=0).values
        self.mean = torch.tensor(mean).float()
        self.variance = torch.tensor(variance).float()
    
    def forward(self, data):
        z, vqloss = self.encoder.forward(data.x_dict, data.edge_index_dict)
        recon_x, edge_probs, zgodnode, foldxout = self.decoder.forward(z, data.x_dict, data.edge_index_dict, None)
        return z, recon_x, edge_probs, zgodnode, foldxout, vqloss
    
    def training_step(self, batch, batch_idx):
        
        batch['Foldx'].x = batch['Foldx'].x.view(-1, 23)
        
        batch['Foldx'].x = self.batchnorm_fx( batch['Foldx'].x )
        
        z,  vqloss = self.encoder(batch.x_dict, batch.edge_index_dict )
        recon_x , edge_probs , zgodnode , foldxout = self.decoder( z, batch.x_dict, batch.edge_index_dict , None ) 
        
        xloss = ft2.aa_reconstruction_loss(batch['AA'].x, recon_x)
        edgeloss = ft2.recon_loss(z, batch.x_dict, batch.edge_index_dict, batch.edge_index_dict[('res', 'contactPoints', 'res')], self.decoder)
        foldxloss = F.smooth_l1_loss( foldxout.view(batch['Foldx'].x.shape) , batch['Foldx'].x )
        loss = 1 * xloss + 0.5 * edgeloss +  vqloss +  foldxloss
        self.log('loss', loss, sync_dist=True , prog_bar=True)
        self.log( 'aa' , xloss  , sync_dist=True ,  prog_bar=True)
        self.log('edge', edgeloss,sync_dist=True ,  prog_bar=True)
        self.log('foldx', foldxloss, sync_dist=True , prog_bar=True)
        self.log('vq', vqloss, sync_dist=True, prog_bar=True)
        #log each loss separately
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate the model
model = GNNModel(ndim, ndim_godnode)
for data in train_loader:
    data = data.to(device)
    model = model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        z,vqloss = model.encoder.forward(data.x_dict , data.edge_index_dict)
        recon_x , edge_probs , zgodnode , foldxout = model.decoder( z, data.x_dict, data.edge_index_dict , None )
    break

# Define Trainer
trainer = pl.Trainer(
    strategy='ddp_find_unused_parameters_true',
    accelerator='gpu',
    devices=-1,  # Use all available GPUs
    max_epochs=1000,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    accumulate_grad_batches=5,
)

# Train the model
trainer.fit(model, train_loader)

# Save the trained model
torch.save(model.state_dict(), 'gnn_model.pt')
