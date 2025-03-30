import foldtree2_ecddcd as ft2
from converter import pdbgraph
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pickle
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

# Create directory for experiment results
experiment_dir = './scaling_experiment/'
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)

# Model configuration
hidden_size = 500  # Single hidden size value
num_epochs = 50
modeldir = './models/'
datadir = '../../datasets/'

# Fixed hyperparameters
batch_size = 10
num_embeddings = 40
embedding_dim = 20
learning_rate = 0.001
edgeweight = 0.1
xweight = 0.1
vqweight = 0.001
foldxweight = 0.001
fapeweight = 0.01
angleweight = 0.01
lddt_weight = 0.1
dist_weight = 0.01
clip_grad = True
ema = True
fapeloss = True  # Enable FAPE loss
lddtloss = False  # Enable LDDT loss
denoise = True    # Enable denoising secondary structure prediction

# Setting the seed for reproducibility
pl.seed_everything(0, workers=True)

# Define Lightning Module for FoldTree
class FoldTreeModule(pl.LightningModule):
    def __init__(self, 
                 hidden_size,
                 ndim, 
                 ndim_godnode, 
                 converter,
                 num_embeddings=40,
                 embedding_dim=20,
                 xweight=0.1,
                 edgeweight=0.1,
                 vqweight=0.001,
                 foldxweight=0.001,
                 fapeweight=0.01,
                 angleweight=0.01,
                 lddt_weight=0.1,
                 dist_weight=0.01,
                 learning_rate=0.001,
                 ema=True,
                 fapeloss=True,
                 lddtloss=True,
                 geometry=False,
                 denoise=True):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.xweight = xweight
        self.edgeweight = edgeweight
        self.vqweight = vqweight
        self.foldxweight = foldxweight
        self.fapeweight = fapeweight
        self.angleweight = angleweight
        self.lddt_weight = lddt_weight
        self.dist_weight = dist_weight
        self.learning_rate = learning_rate
        self.fapeloss = fapeloss
        self.lddtloss = lddtloss
        self.geometry = geometry
        self.denoise = denoise
        
        # Create encoder
        encoder_layers = 2
        self.encoder = ft2.mk1_Encoder(
            in_channels=ndim, 
            hidden_channels=[hidden_size] * encoder_layers,
            out_channels=embedding_dim, 
            metadata={'edge_types': [
                ('res','contactPoints', 'res'), 
                ('res','backbone', 'res'),
                ('res','backbonerev', 'res')
            ]},
            num_embeddings=num_embeddings, 
            commitment_cost=0.9, 
            edge_dim=1,
            encoder_hidden=hidden_size * 4, 
            EMA=ema, 
            nheads=10, 
            dropout_p=0.005,
            reset_codes=False, 
            flavor='transformer'
        )
        
        # Create decoder
        decoder_layers = 3
        self.decoder = ft2.HeteroGAE_Decoder(
            in_channels={
                'res': self.encoder.out_channels, 
                'godnode4decoder': ndim_godnode,
                'foldx': 23
            },
            hidden_channels={
                ('res', 'window', 'res'): [hidden_size] * decoder_layers,
                ('res', 'backbone', 'res'): [hidden_size] * decoder_layers,
                ('res', 'backbonerev', 'res'): [hidden_size] * decoder_layers,
                ('res', 'informs', 'godnode4decoder'): [hidden_size] * decoder_layers,
            },
            layers=decoder_layers,
            metadata=converter.metadata,
            amino_mapper=converter.aaindex,
            concat_positions=False,
            flavor='sage',
            output_foldx=True,
            geometry=self.geometry,  # Enable geometry prediction based on parameter
            denoise=self.denoise,
            Xdecoder_hidden=[hidden_size, hidden_size//2, hidden_size//5],
            PINNdecoder_hidden=[hidden_size//2, hidden_size//4, hidden_size//5],
            geodecoder_hidden=[hidden_size//3, hidden_size//3, hidden_size//3],
            AAdecoder_hidden=[hidden_size, hidden_size//5, hidden_size//10],
            contactdecoder_hidden=[hidden_size//2, hidden_size//4],
            nheads=10,
            dropout=0.005,
            residual=False,
            normalize=True,
            contact_mlp=False
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Flag to track initialization
        self.initialized = False
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, data):
        z, vqloss = self.encoder.forward(data)
        data['res'].x = z
        recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2 = self.decoder(data, None)
        return z, vqloss, recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2
    
    def training_step(self, batch, batch_idx):
        # Move batch to current device
        data = batch.to(self.device)
        
        # Initialize lazy modules if not done yet
        if not self.initialized:
            with torch.no_grad():
                z, vqloss = self.encoder.forward(data)
                data['res'].x = z
                recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2 = self.decoder(data, None)
                self.initialized = True
                # Return None for the first step to skip the backward pass
                return None
        
        # Forward pass
        z, vqloss = self.encoder.forward(data)
        data['res'].x = z
        edgeloss, distloss = ft2.recon_loss(data, data.edge_index_dict[('res', 'contactPoints', 'res')], 
                                           self.decoder, plddt=False, offdiag=False)
        recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2 = self.decoder(data, None)
        
        # Compute amino acid reconstruction loss
        xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
        
        # Compute foldx loss if available
        if hasattr(self.decoder, 'output_foldx') and self.decoder.output_foldx and 'Foldx' in data:
            data['Foldx'].x = data['Foldx'].x.view(-1, 23)
            if hasattr(self.decoder, 'bn_foldx'):
                data['Foldx'].x = self.decoder.bn_foldx(data['Foldx'].x)
            foldxout = foldxout.view(data['Foldx'].x.shape)
            foldxloss = F.smooth_l1_loss(foldxout, data['Foldx'].x)
        else:
            foldxloss = torch.tensor(0.0, device=self.device)
            
        # Compute geometric losses
        if self.geometry:
            # Angle loss with reduction='none' first
            angleloss = F.smooth_l1_loss(angles, data.x_dict['bondangles'], reduction='none')
            fapeloss = torch.tensor(0.0, device=self.device)
            lddtloss = torch.tensor(0.0, device=self.device)
            
            # Additional calculations if denoise is enabled
            if self.denoise:
                # Add second angle loss from denoised prediction
                angleloss += F.smooth_l1_loss(angles2, data.x_dict['bondangles'], reduction='none')
                
                # FAPE loss using the denoised predictions (r2, t2)
                if self.fapeloss and 't_true' in data and 'R_true' in data:
                    batch_data = data['t_true'].batch if hasattr(data['t_true'], 'batch') else None
                    t_true = data['t_true'].x
                    R_true = data['R_true'].x
                    
                    # Use r2, t2 for the denoised predictions
                    fapeloss = ft2.fape_loss(
                        true_R=R_true,
                        true_t=t_true,
                        pred_R=r2,
                        pred_t=t2,
                        batch=batch_data,
                        d_clamp=10.0,
                        eps=1e-8,
                        plddt=None,
                        soft=False
                    )
                
                # LDDT loss using the coordinates directly
                if self.lddtloss and hasattr(ft2, 'lddt_loss') and 'coords' in data:
                    batch_data = data['t_true'].batch if hasattr(data['t_true'], 'batch') else None
                    lddtloss = ft2.lddt_loss(
                        coord_true=data['coords'].x,
                        pred_R=r,
                        pred_t=t,
                        batch=batch_data
                    )
            
            # Take the mean of angle loss at the end
            angleloss = angleloss.mean()
        else:
            # Set all geometric losses to zero if geometry is disabled
            angleloss = torch.tensor(0.0, device=self.device)
            fapeloss = torch.tensor(0.0, device=self.device)
            lddtloss = torch.tensor(0.0, device=self.device)
        
        # Compute total loss
        total_loss = (
            self.xweight * xloss + 
            self.edgeweight * edgeloss + 
            self.vqweight * vqloss +
            self.foldxweight * foldxloss +
            self.fapeweight * fapeloss +
            self.angleweight * angleloss +
            self.lddt_weight * lddtloss +
            self.dist_weight * distloss
        )
        
        # Log metrics
        self.log('Loss/AA', xloss.item(), prog_bar=True, sync_dist=True)
        self.log('Loss/Edge', edgeloss.item(), sync_dist=True)
        self.log('Loss/VQ', vqloss.item(), sync_dist=True)
        self.log('Loss/Foldx', foldxloss.item(), sync_dist=True)
        self.log('Loss/FAPE', fapeloss.item(), sync_dist=True)
        self.log('Loss/Angle', angleloss.item(), sync_dist=True)
        self.log('Loss/LDDT', lddtloss.item(), sync_dist=True)
        self.log('Loss/Distance', distloss.item(), sync_dist=True)
        self.log('Loss/Total', total_loss.item(), prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Loss/Total',
                'interval': 'epoch'
            }
        }
        
    def on_save_checkpoint(self, checkpoint):
        # Save encoder and decoder separately as pickle file for compatibility
        model_path = f"{experiment_dir}/model_hidden{self.hidden_size}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump((self.encoder, self.decoder), f)

# Load the full dataset
print("Loading dataset...")
full_dataset = pdbgraph.StructureDataset('structs_training_godnodemk5.h5')
converter = pdbgraph.PDB2PyG()

# Get a data sample to determine dimensions
sample_loader = DataLoader(full_dataset, batch_size=1)
data_sample = next(iter(sample_loader))
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]

print(f"Dataset loaded with {len(full_dataset)} samples")
print(f"Using ndim={ndim}, ndim_godnode={ndim_godnode}")

# Create the model
model = FoldTreeModule(
    hidden_size=hidden_size,
    ndim=ndim,
    ndim_godnode=ndim_godnode,
    converter=converter,
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    xweight=xweight,
    edgeweight=edgeweight,
    vqweight=vqweight,
    foldxweight=foldxweight,
    fapeweight=fapeweight,
    angleweight=angleweight,
    lddt_weight=lddt_weight,
    dist_weight=dist_weight,
    learning_rate=learning_rate,
    ema=ema,
    fapeloss=fapeloss,
    lddtloss=lddtloss,
    geometry=True,  # Enable geometry prediction to use angle loss, FAPE, etc.
    denoise=denoise
)

# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} parameters")

# Create dataloader
train_loader = DataLoader(
    full_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    worker_init_fn=np.random.seed(0),
    num_workers=6
)

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{experiment_dir}/checkpoints",
    filename="model-{epoch:02d}-{Loss/Total:.4f}",
    monitor="Loss/Total",
    mode="min",
    save_top_k=1
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
progress_bar = TQDMProgressBar()

# Setup logger
logger = TensorBoardLogger(
    save_dir=experiment_dir,
    name="logs",
    default_hp_metric=False
)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="gpu",
    devices="auto",  # Use all available GPUs
    strategy="ddp",  # Distributed Data Parallel
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor, progress_bar],
    gradient_clip_val=1.0 if clip_grad else None,
)

# Log hyperparameters
logger.log_hyperparams({
    "hidden_size": hidden_size,
    "num_embeddings": num_embeddings,
    "embedding_dim": embedding_dim,
    "learning_rate": learning_rate,
    "edgeweight": edgeweight,
    "xweight": xweight,
    "vqweight": vqweight,
    "foldxweight": foldxweight,
    "fapeweight": fapeweight,
    "angleweight": angleweight,
    "lddt_weight": lddt_weight,
    "dist_weight": dist_weight,
    "fapeloss": fapeloss,
    "lddtloss": lddtloss,
    "denoise": denoise,  # Add denoise to logged hyperparameters
    "batch_size": batch_size,
    "num_parameters": num_params,
})

if __name__ == "__main__":
    # Train the model
    print(f"Starting training on {trainer.num_devices} GPUs")
    trainer.fit(model, train_loader)

    # Save final model
    final_model_path = f"{experiment_dir}/final_model.pkl"
    with open(final_model_path, 'wb') as f:
        pickle.dump((model.encoder, model.decoder), f)
    
    print(f"Training complete. Final model saved to {final_model_path}")
