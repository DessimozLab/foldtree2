import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import foldtree2_ecddcd as ft2
from converter import pdbgraph
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import numpy as np
import glob
import torch
import torch.nn.functional as F
from torch.optim import Adam
from converter import pdbgraph
from torch_geometric.data import DataLoader
import pickle
import src.losses.fafe as fafe
import pandas as pd
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter
# foldtree2 / local imports

# Some of your original constants / device check
AVAIL_GPUS = min(1, torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------- #
# Example data setup (adjust as needed)
# ---------------------------------------------------------------------------- #

datadir = '../../datasets/'
pdbfiles = glob.glob(datadir +'structs/*.pdb')

converter = pdbgraph.PDB2PyG()
# Example structure:
data_sample = converter.struct2pyg(pdbfiles[0], foldxdir='./foldx/', verbose=False)
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]

# Example: load precomputed dataset (adjust to your h5 file as needed)
struct_dat = pdbgraph.StructureDataset('structs_training_godnodemk4.h5')

# DataLoader
batch_size = 20
train_loader = DataLoader(struct_dat, batch_size=batch_size, shuffle=True, num_workers=6)

# Mean/variance for foldx (if you use them)
mean = pd.read_csv('foldxmean.csv', index_col=0)
variance = pd.read_csv('foldxvariance.csv', index_col=0)
mean = torch.tensor(mean.values).float()
variance = torch.tensor(variance.values).float()

# ---------------------------------------------------------------------------- #
# Create your models: encoder and decoder
# ---------------------------------------------------------------------------- #

encoder_layers = 2
decoder_layers = 4

encoder = ft2.mk1_Encoder(
    in_channels=ndim,
    hidden_channels=[100]*encoder_layers,
    out_channels=20,
    metadata=converter.metadata,
    num_embeddings=40,
    commitment_cost=0.9,
    edge_dim=1,
    encoder_hidden=300,
    EMA=True,
    nheads=4,
    dropout_p=0.001,
    reset_codes=False,
    flavor='gat'
)

decoder = ft2.HeteroGAE_Decoder(
    in_channels={'res': encoder.out_channels + 256,
                 'godnode4decoder': ndim_godnode,
                 'foldx': 23},
    hidden_channels={
        ('res', 'informs', 'godnode4decoder'): [75]*decoder_layers,
        ('godnode4decoder', 'informs', 'res'): [75]*decoder_layers,
        ('res', 'backbone', 'res'): [75]*decoder_layers,
        ('res', 'backbonerev', 'res'): [75]*decoder_layers,
    },
    layers=decoder_layers,
    metadata=converter.metadata,
    amino_mapper=converter.aaindex,
    flavor='sage',
    output_foldx=True,
    contact_mlp=True,
    denoise=True,
    Xdecoder_hidden=250,
    PINNdecoder_hidden=[100, 50, 10],
    contactdecoder_hidden=[50, 50],
    nheads=4,
    dropout=0.001,
    AAdecoder_hidden=[100, 100, 20],
)

# ---------------------------------------------------------------------------- #
# LightningModule
# ---------------------------------------------------------------------------- #

class LitModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        fapeloss=True,
        learning_rate=1e-3,
        edgeweight=1.0,
        xweight=1.0,
        vqweight=0.1,
        foldxweight=0.01,
        fapeweight=0.1,
        angleweight=0.1,
        err_eps=1e-2
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])  # keep hyperparams track
        self.encoder = encoder
        self.decoder = decoder
        
        self.fapeloss = fapeloss
        self.learning_rate = learning_rate
        
        # Weights for each loss term
        self.edgeweight = edgeweight
        self.xweight = xweight
        self.vqweight = vqweight
        self.foldxweight = foldxweight
        self.fapeweight = fapeweight
        self.angleweight = angleweight
        self.err_eps = err_eps
        
        # Example placeholders if you rely on external mean/var
        # (You can pass them or define them here.)
        # self.mean = ...
        # self.variance = ...
        
        # If needed, you can define some trackable metrics
        self.train_aa_loss = 0.0
        self.train_edge_loss = 0.0
        self.train_vq_loss = 0.0
        self.train_foldx_loss = 0.0
        self.train_fape_loss = 0.0
        self.train_angle_loss = 0.0

        # Any other buffers or modules can go here
        # e.g. self.register_buffer('some_mean', torch.zeros(1))

    def forward(self, data):
        """
        Forward pass = encoder + concatenation with positions + feeding to decoder.
        Typically you won't call this directly in training_step except for convenience.
        """
        z, vq_loss = self.encoder.forward(data)
        # Append positional encoding from data.x_dict['positions'] if present
        if 'positions' in data.x_dict:
            z = torch.cat((z, data.x_dict['positions']), dim=1)
        data['res'].x = z
        
        # Run forward through decoder
        recon_x, edge_probs, zgodnode, foldxout, r, t, angles = self.decoder(data, None)
        return recon_x, edge_probs, vq_loss, foldxout, r, t, angles

    def training_step(self, batch, batch_idx):
        # Forward
        recon_x, edge_probs, vq_loss, foldxout, R_pred, t_pred, angles = self(batch)

        # 1) Edge reconstruction loss (if you have a special function, e.g. `ft2.recon_loss`)
        edgeloss = ft2.recon_loss(
            batch,
            batch.edge_index_dict[('res', 'contactPoints', 'res')],
            self.decoder,
            distweight=True
        )

        # 2) AA reconstruction loss
        xloss = ft2.aa_reconstruction_loss(batch['AA'].x, recon_x)

        # 3) FoldX output loss
        if self.decoder.output_foldx:
            batch['Foldx'].x = batch['Foldx'].x.view(-1, 23)
            # If your decoder has a batch norm on foldx:
            batch['Foldx'].x = self.decoder.bn_foldx(batch['Foldx'].x)
            foldxout = foldxout.view(batch['Foldx'].x.shape)
            foldxloss = F.smooth_l1_loss(foldxout, batch['Foldx'].x)
        else:
            foldxloss = torch.tensor(0.0, device=self.device)

        # 4) FAPE loss
        if self.fapeloss:
            batch_data = batch['t_true'].batch  # see how your dataset organizes
            t_true = batch['t_true'].x
            R_true = batch['R_true'].x
            # Example using the frame_aligned_frame_error or your custom function:
            fploss = ft2.fape_loss(
                true_R=R_true,
                true_t=t_true,
                pred_R=R_pred,
                pred_t=t_pred,
                batch=batch_data,
                d_clamp=10.0,
                eps=1e-8,
                plddt=batch['plddt'].x,
                soft=False
            )
        else:
            fploss = torch.tensor(0.0, device=self.device)

        # 5) Angle loss
        angleloss = F.smooth_l1_loss(
            angles * batch['plddt'].x,
            batch.x_dict['bondangles'] * batch['plddt'].x
        )

        # Combine losses
        loss = (self.xweight * xloss +
                self.edgeweight * edgeloss +
                self.vqweight * vq_loss +
                self.foldxweight * foldxloss +
                self.fapeweight * fploss +
                self.angleweight * angleloss)

        # Logging (by default logs go to "loss" for your progress bar or to TensorBoard)
        # log(..., on_epoch=True) to aggregate and produce an epoch-level metric
        self.log('Loss/Train', loss, on_step=False, on_epoch=True)
        self.log('Loss/AA', xloss, on_step=False, on_epoch=True)
        self.log('Loss/Edge', edgeloss, on_step=False, on_epoch=True)
        self.log('Loss/VQ', vq_loss, on_step=False, on_epoch=True)
        self.log('Loss/Foldx', foldxloss, on_step=False, on_epoch=True)
        self.log('Loss/Fape', fploss, on_step=False, on_epoch=True)
        self.log('Loss/Angle', angleloss, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """
        If you want to replicate some of your dynamic weighting or
        checkpoint saving logic at the end of each epoch, do it here.
        For example:
        
        epoch = self.current_epoch
        if epoch % 10 == 0:
            # Save a pkl of (encoder, decoder)? Typically we rely on
            # Trainer checkpoint callbacks in Lightning, but you can do manual saves too:
            with open(f'checkpoint_{epoch}.pkl', 'wb') as f:
                pickle.dump((self.encoder, self.decoder), f)
                
        # Optionally replicate logic to adjust xweight, foldxweight, etc. 
        """
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        # Return format that monitors a particular logged metric (e.g. Loss/Train):
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Loss/Train'  # or 'Loss/AA', etc.
            }
        }

# ---------------------------------------------------------------------------- #
# Training via PyTorch Lightning Trainer
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    # Instantiate the LightningModule
    model = LitModel(
        encoder=encoder,
        decoder=decoder,
        fapeloss=True,
        learning_rate=1e-3,
        edgeweight=1.0,
        xweight=1.0,
        vqweight=0.1,
        foldxweight=0.01,
        fapeweight=0.1,
        angleweight=0.1,
        err_eps=1e-2
    )


    #init lazy modules 
    with torch.no_grad():  # Initialize lazy modules.
        data = next(iter(train_loader))
        z,vqloss = model.encoder.forward(data)
        z = torch.cat( (z, data.x_dict['positions'] ) , dim = 1)
        data['res'].x = z
        recon_x , edge_probs , zgodnode , foldxout, r , t , angles = model.decoder(  data , None )


    # Create a trainer
    trainer = pl.Trainer(
        max_epochs=800,
        #save the best model
        callbacks=[pl.callbacks.ModelCheckpoint( monitor='Loss/Train', mode='min', save_top_k=1, dirpath='.', filename='best_model')],
    )
    
    # Fit / train
    trainer.fit(model, train_loader)
    
    # Save final model states if you want:
    torch.save(model.encoder.state_dict(), 'encoder_final.ckpt')
    torch.save(model.decoder.state_dict(), 'decoder_final.ckpt')
    
    # If you want to pickle them together:
    with open('final_model.pkl', 'wb') as f:
        pickle.dump((model.encoder, model.decoder), f)
