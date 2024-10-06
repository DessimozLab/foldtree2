import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
import pickle
import os
import tqdm
import foldtree2_ecddcd as ft2

# PyTorch Lightning module
class GAEModel(pl.LightningModule):
    def __init__(self, encoder, decoder, variational=False, betafactor=2, lr=0.01):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.variational = variational
        self.betafactor = betafactor
        self.lr = lr
        self.edgeweight = .05
        self.xweight = 1
        self.vqweight = 1
        self.plddtweight = 1

    def forward(self, data):
        if self.variational:
            z, mu, logvar, vqloss, eloss, qloss = self.encoder.forward(data['res'].x, data['AA'].x, data.edge_index_dict)
            return z, vqloss, eloss, qloss, mu, logvar
        else:
            z, vqloss, eloss, qloss = self.encoder.forward(data['res'].x, data['AA'].x, data.edge_index_dict)
            return z, vqloss, eloss, qloss

    def training_step(self, batch, batch_idx):
        z, vqloss, eloss, qloss = self(batch)
        
        edgeloss = ft2.recon_loss(z, batch.edge_index_dict[('res', 'contactPoints', 'res')],
                                  batch.edge_index_dict[('res', 'backbone', 'res')], self.decoder)
        
        decode_out = self.decoder(z, batch.edge_index_dict[('res', 'contactPoints', 'res')],
                                  batch.edge_index_dict, poslossmod=1, neglossmod=1)
        xloss = ft2.aa_reconstruction_loss(batch['AA'].x, decode_out[0])

        loss = self.xweight * xloss + self.edgeweight * edgeloss + self.vqweight * vqloss
        
        if self.variational:
            mu, logvar = eloss, qloss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl_loss * self.betafactor

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        return optimizer

# Load dataset
converter = ft2.PDB2PyG()
struct_dat = ft2.StructureDataset('structs_training.h5')

# Model parameters
ndim = struct_dat[0]['res'].x.shape[1]

encoder = ft2.HeteroGAE_Encoder(in_channels={'res':ndim}, hidden_channels=[400, 400, 400 ] , 
                        out_channels=250, metadata=converter.metadata , num_embeddings=64, 
                        commitment_cost= .8 , encoder_hidden= 400  , nheads = 4 , average = False
                        , reset_codes= False , dropout_p=0.001 , separated = separated , flavor = 'sage' )

decoder = ft2.HeteroGAE_Decoder(encoder_out_channels = encoder.out_channels , 
                            hidden_channels={ ( 'res','backbone','res'):[  100 ] * 5  }   , 
                            out_channels_hidden= 200 , metadata=converter.metadata , 
                            amino_mapper = converter.aaindex 
                            , Xdecoder_hidden= 100 , nheads = 4 , dropout = 0.001  , AAdecoder_hidden = [100,50,25] , flavor = 'SAGE' )  


# Load saved model if available
model_dir = 'models/'
encoder_save = 'encoder_lowcost_lightning'
decoder_save = 'decoder_lowcost_lightning'

if os.path.exists(model_dir + encoder_save):
    encoder.load_state_dict(torch.load(model_dir + encoder_save))
    print('Loaded encoder')

if os.path.exists(model_dir + decoder_save):
    decoder.load_state_dict(torch.load(model_dir + decoder_save))
    print('Loaded decoder')

# Training
train_loader = DataLoader(struct_dat, batch_size=40, shuffle=True)

# PyTorch Lightning trainer
model = GAEModel(encoder=encoder, decoder=decoder, variational=False, betafactor=2, lr=0.001)

trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", devices=2, strategy="auto")
trainer.fit(model, train_loader)