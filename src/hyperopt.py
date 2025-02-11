

import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pickle
import os
import numpy as np
import tqdm

import optuna
from optuna.trial import TrialState

import foldtree2_ecddcd as ft2

converter = ft2.PDB2PyG()

encoder_save = 'encoder_mk3_aa_EMA_64_lowcost'
decoder_save = 'decoder_mk3_aa_EMA_64_lowcost'
overwrite = False
train_loop = True


struct_dat = ft2.StructureDataset('structs_training.h5')

# Create a DataLoader for training
total_loss_x = 0
total_loss_edge = 0
total_vq=0
total_kl = 0
total_plddt=0
# Training loop


ndim = struct_dat[0]['res'].x.shape[1]
print( struct_dat[0] )

#load model if it exists

#add positional encoder channels to input
encoder = ft2.HeteroGAE_Encoder(in_channels=ndim, hidden_channels=[ 400 ]*3 , out_channels=250, metadata=converter.metadata , num_embeddings=64, commitment_cost=.9 , encoder_hidden=500 , EMA = True , reset_codes= False )
#encoder = HeteroGAE_VariationalQuantizedEncoder(in_channels=ndim, hidden_channels=[100]*3 , out_channels=25, metadata=metadata , num_embeddings=256  , commitment_cost= 1.5 )

decoder = ft2.HeteroGAE_Decoder(encoder_out_channels = encoder.out_channels , 
                            hidden_channels={ ( 'res','backbone','res'):[ 500 ] * 5  } , 
                            out_channels_hidden= 150 , metadata=converter.metadata , amino_mapper = converter.aaindex , Xdecoder_hidden=100 )




if os.path.exists(encoder_save) and overwrite == False:
    encoder.load_state_dict(torch.load(encoder_save ))
if os.path.exists(decoder_save) and overwrite == False:
    decoder.load_state_dict(torch.load(decoder_save  ))

#create a training loop for the GAE model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')
print(device)


betafactor = 2
#put encoder and decoder on the device
encoder = encoder.to(device)
decoder = decoder.to(device)




# Define the training function
def train_model(model, optimizer, dataset, epochs=20):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        for data in dataset:
            data = data.to(device)
            z_quantized, vq_loss, mu, logvar = model.encoder(data.x, data.edge_index_dict)
            x_reconstructed, edge_probs = model.decoder(z_quantized, data.edge_index, backbone)
            edge_recon_loss = recon_loss(z_quantized, data['res'].edge_index, backbone, model.decoder)
            x_loss = x_reconstruction_loss(data['AA'].x, x_reconstructed)
            total_loss = edge_recon_loss + x_loss + vq_loss
            total_loss.backward()
            optimizer.step()
    return total_loss.item()


# Define the objective function for Optuna
def objective(trial , dataset=struct_dat):
    in_channels = ndim


    encoder_hidden_channels = trial.suggest_categorical('hidden_channels', [400, 600, 800])
    
    decoder_hidden_channels = trial.suggest_categorical('hidden_channels', [400, 600, 800])
    
    encoder_hidden_convs = trial.suggest_categorical('hidden_channels', [3,4,5])
    
    decoder_hidden_convs = trial.suggest_categorical('hidden_channels', [4,5,6])
    
    latent_dim_encoder = trial.suggest_int('latent_dim', 128, 512)

    latent_dim_decoder = trial.suggest_int('latent_dim', 128, 512)

    latent_dim_xdecoder = trial.suggest_int('latent_dim', 50, 250)

    encoding_dim = trial.suggest_int('latent_dim', 128, 512)

    num_embeddings = trial.suggest_int('num_embeddings', 20, 64)

    commitment_cost = trial.suggest_float('commitment_cost', 0.1, 1.0)

    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    
    model = nn.Module()

    model.encoder = ft2.HeteroGAE_Encoder(in_channels=ndim, hidden_channels=[encoder_hidden_channels ]*encoder_hidden_convs , out_channels=encoding_dim, 
    
    metadata=converter.metadata , num_embeddings=num_embeddings, commitment_cost=commitment_cost , encoder_hidden=latent_dim_encoder , EMA = True , reset_codes= False )
    
    model.decoder = ft2.HeteroGAE_Decoder(encoder_out_channels = encoder.out_channels , 
                            hidden_channels={ ( 'res','backbone','res'):[ decoder_hidden_channels ] * decoder_hidden_convs  } , 
                            out_channels_hidden=latent_dim_decoder , metadata=converter.metadata , amino_mapper = converter.aaindex , Xdecoder_hidden=latent_dim_xdecoder )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Dummy data and backbone for illustration purposes
    loss = train_model(model, optimizer, dataset )
    return loss

#if study exists, load it
if os.path.exists('study.pkl'):
    with open('study.pkl', 'rb') as f:
        study = pickle.load(f)
else:
    # Create the study
    study = optuna.create_study(direction='minimize')

# Optimize the study
study.optimize(objective, n_trials=50)
# Save the study
with open('study.pkl', 'wb') as f:
    pickle.dump(study, f)



# Get the best trial
best_trial = study.best_trial
print(f'Best trial: {best_trial.params}')
# Save the best trial
with open('best_trial.pkl', 'wb') as f:
    pickle.dump(best_trial, f)