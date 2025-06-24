import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from src import foldtree2_ecddcd as ft2
from src import pdbgraph
from src.losses import fafe as fafe
import numpy as np
import pandas as pd
import os
import argparse
import pickle
import sys

# Argument parsing (copied from learn.py)
parser = argparse.ArgumentParser(description='Train model with configurable parameters (scaling_experiment style)')
parser.add_argument('--dataset', '-d', type=str, default='structs_training_godnodemk5.h5',
                    help='Path to the dataset file (default: structs_training_godnodemk5.h5)')
parser.add_argument('--hidden-size', '-hs', type=int, default=100,
                    help='Hidden layer size (default: 100)')
parser.add_argument('--epochs', '-e', type=int, default=800,
                    help='Number of epochs for training (default: 800)')
parser.add_argument('--device', type=str, default=None,
                    help='Device to run on (e.g., cuda:0, cuda:1, cpu) (default: auto-select)')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.00001,
                    help='Learning rate (default: 0.00001)')
parser.add_argument('--batch-size', '-bs', type=int, default=20,
                    help='Batch size (default: 20)')
parser.add_argument('--output-dir', '-o', type=str, default='./models/',
                    help='Directory to save models/results (default: ./models/)')
parser.add_argument('--model-name', type=str, default='newmodelmk6tanh',
                    help='Model name for saving (default: newmodelmk6tanh)')
parser.add_argument('--num-embeddings', type=int, default=40,
                    help='Number of embeddings for the encoder (default: 40)')
parser.add_argument('--embedding-dim', type=int, default=20,
                    help='Embedding dimension for the encoder (default: 20)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite saved model if exists, otherwise continue training')
parser.add_argument('--geometry', action='store_true', help='Train the model with geometry')
parser.add_argument('--fapeloss', action='store_true', help='Train the model with FAPE loss')
parser.add_argument('--lddtloss', action='store_true', help='Train the model with LDDT loss')
parser.add_argument('--concat-positions', action='store_true', help='Train the model with positional encoding')
parser.add_argument('--transformer', action='store_true', help='Train the model with transformer decoder')
parser.add_argument('--output-foldx', action='store_true', help='Train the model with foldx output')

if len(sys.argv) == 1:
    print('No arguments provided. Use -h for help.')
    print('Example command: python learn_lightning.py -d structs_training_godnodemk5.h5 -o ./models/ -lr 0.0001 -e 800 -bs 20 --geometry --fapeloss --lddtloss --concat-positions --transformer')
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()

# Set up device
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data and model setup
datadir = '../../datasets/foldtree2/'
converter = pdbgraph.PDB2PyG(aapropcsv='src/aaindex1.csv')
struct_dat = pdbgraph.StructureDataset(args.dataset)
train_loader = DataLoader(struct_dat, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=6)
data_sample = next(iter(train_loader))
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]

# Model directory
os.makedirs(args.output_dir, exist_ok=True)
model_path = os.path.join(args.output_dir, args.model_name + '.pkl')

# Model (load or create)
if os.path.exists(model_path) and not args.overwrite:
    with open(model_path, 'rb') as f:
        encoder, decoder = pickle.load(f)
else:
    encoder_layers = 2
    encoder = ft2.mk1_Encoder(
        in_channels=ndim,
        hidden_channels=[args.hidden_size] * encoder_layers,
        out_channels=args.embedding_dim,
        metadata={'edge_types': [('res', 'contactPoints', 'res'), ('res', 'backbone', 'res'), ('res', 'backbonerev', 'res')]},
        num_embeddings=args.num_embeddings,
        commitment_cost=.9,
        edge_dim=1,
        encoder_hidden=args.hidden_size * 4,
        EMA=True,
        nheads=10,
        dropout_p=0.005,
        reset_codes=False,
        flavor='transformer' if args.transformer else 'sage'
    )
    decoder_layers = 5
    decoder = ft2.HeteroGAE_Decoder(
        in_channels={'res': encoder.out_channels, 'godnode4decoder': ndim_godnode, 'foldx': 23},
        hidden_channels={
            ('res', 'backbone', 'res'): [args.hidden_size] * decoder_layers,
            ('res', 'backbonerev', 'res'): [args.hidden_size] * decoder_layers,
            ('res', 'informs', 'godnode4decoder'): [args.hidden_size] * decoder_layers,
        },
        layers=decoder_layers,
        metadata=converter.metadata,
        amino_mapper=converter.aaindex,
        concat_positions=args.concat_positions or args.transformer,
        flavor='sage',
        output_foldx=args.output_foldx,
        Xdecoder_hidden=[args.hidden_size, args.hidden_size // 2, max(1, args.hidden_size // 5)],
        PINNdecoder_hidden=[max(1, args.hidden_size // 2), max(1, args.hidden_size // 4), max(1, args.hidden_size // 5)],
        AAdecoder_hidden=[args.hidden_size, args.hidden_size // 2, args.hidden_size // 2],
        contactdecoder_hidden=[max(1, args.hidden_size // 2), max(1, args.hidden_size // 4)],
        nheads=10,
        dropout=0.005,
        residual=False,
        normalize=True,
        contact_mlp=True
    )

# LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'args'])

        # Loss weights (copied from learn.py)
        self.edgeweight = .01
        self.xweight = .1
        self.vqweight = .0001
        self.foldxweight = .001
        self.fapeweight = .001
        self.angleweight = .001
        self.lddt_weight = .1
        self.dist_weight = .01

    def forward(self, data):
        z, vqloss = self.encoder.forward(data)
        data['res'].x = z
        recon_x, edge_probs, zgodnode, foldxout = self.decoder(data, None)
        return recon_x, edge_probs, vqloss, foldxout

    def training_step(self, batch, batch_idx):
        z, vqloss = self.encoder.forward(batch)
        batch['res'].x = z
        edgeloss, distloss = ft2.recon_loss(batch, batch.edge_index_dict[('res', 'contactPoints', 'res')], self.decoder, plddt=False, offdiag=False)
        recon_x, edge_probs, zgodnode, foldxout = self.decoder(batch, None)
        xloss = ft2.aa_reconstruction_loss(batch['AA'].x, recon_x)
        if self.decoder.output_foldx:
            batch['Foldx'].x = batch['Foldx'].x.view(-1, 23)
            batch['Foldx'].x = self.decoder.bn_foldx(batch['Foldx'].x)
            foldxout = foldxout.view(batch['Foldx'].x.shape)
            foldxloss = F.smooth_l1_loss(foldxout, batch['Foldx'].x)
        else:
            foldxloss = torch.tensor(0.0, device=self.device)
        # Geometry, FAPE, LDDT, angle losses (dummy for now)
        fploss = torch.tensor(0.0, device=self.device)
        lddt_loss = torch.tensor(0.0, device=self.device)
        angleloss = torch.tensor(0.0, device=self.device)
        # Combine losses
        loss = (self.xweight * xloss +
                self.edgeweight * edgeloss +
                self.vqweight * vqloss +
                self.foldxweight * foldxloss +
                self.fapeweight * fploss +
                self.angleweight * angleloss +
                self.lddt_weight * lddt_loss)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True)
        self.log('Loss/AA', xloss, on_step=False, on_epoch=True)
        self.log('Loss/Edge', edgeloss, on_step=False, on_epoch=True)
        self.log('Loss/VQ', vqloss, on_step=False, on_epoch=True)
        self.log('Loss/Foldx', foldxloss, on_step=False, on_epoch=True)
        self.log('Loss/Fape', fploss, on_step=False, on_epoch=True)
        self.log('Loss/Angle', angleloss, on_step=False, on_epoch=True)
        self.log('Loss/LDDT', lddt_loss, on_step=False, on_epoch=True)
        self.log('Loss/Dist', distloss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Loss/Train'
            }
        }

# Lightning Trainer
if __name__ == '__main__':
    model = LitModel(encoder, decoder, args)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='Loss/Train', mode='min', save_top_k=1, dirpath=args.output_dir, filename=args.model_name)],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        strategy='auto'
    )
    trainer.fit(model, train_loader)
    # Save final model
    with open(model_path, 'wb') as f:
        pickle.dump((model.encoder, model.decoder), f)
