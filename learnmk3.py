import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pickle
import os
import numpy as np
import tqdm

import foldtree2_ecddcd as ft2

converter = ft2.PDB2PyG()

encoder_save = 'encoder_mk2_aa_EMA_248'
decoder_save = 'decoder_mk2_aa_EMA_248'
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
encoder = ft2.HeteroGAE_Encoder(in_channels=ndim, hidden_channels=[ 400 ]*3 , out_channels=250, metadata=converter.metadata , num_embeddings=248, commitment_cost=1 , encoder_hidden=500 , EMA = True )
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


if train_loop == True:
    train_loader = DataLoader(struct_dat, batch_size=40, shuffle=True)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    encoder.train()
    decoder.train()
    
    xlosses = []
    edgelosses = []
    vqlosses = []
    plddtlosses = []

    edgeweight = 2
    xweight = 2
    vqweight = 1
    plddtweight = 1

    for epoch in range(1000):
        for data in tqdm.tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            z,vqloss = encoder.forward(data['res'].x, data['AA'].x , data.edge_index_dict)
            
            #add positional encoding to give to the decoder
            edgeloss = ft2.recon_loss(z , data.edge_index_dict[( 'res','contactPoints','res')]
                                  , data.edge_index_dict[( 'res','backbone','res')], decoder)
            
            recon_x , edge_probs = decoder(z, data.edge_index_dict[( 'res','contactPoints','res')] , data.edge_index_dict )        
    
            xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
            #plddtloss = x_reconstruction_loss(data['plddt'].x, recon_plddt)
            loss = xweight*xloss + edgeweight*edgeloss + vqweight*vqloss #+ plddtloss
            loss.backward()
            optimizer.step()
            total_loss_edge += edgeloss.item()
            total_loss_x += xloss.item()
            total_vq += vqloss.item()
            #total_plddt += plddtloss.item()

        if epoch % 100 == 0 :
            #save model
            torch.save(encoder.state_dict(), encoder_save)
            torch.save(decoder.state_dict(), decoder_save)
        """
        for loss in [( total_loss_x , xlosses , xweight ), (total_loss_edge, edgelosses, edgeweight), ( total_vq, vqlosses, vqweight ) ]:
            loss[1].append(loss[0])
            #calculate the mean delta loss for past 10 epochs
            if len(loss[1]) > 10:
                loss[1].pop(0)
                mean_loss = np.mean(loss[0:5])
                #calculate the delta loss for the last 5 epochs
                delta_loss = np.mean(loss[1][-5:])
                delta_loss = delta_loss- mean_loss
                if delta_loss > 0:
                    loss[2] = loss[2]*2
                else:
                    loss[2] = loss[2]/1.5
                loss[2] = np.clip(loss[2], 1e-5, 1e5)
        """    
        print(f'Epoch {epoch}, AALoss: {total_loss_x:.4f}, Edge Loss: {total_loss_edge:.4f}, vq Loss: {total_vq:.4f}') #, plddt Loss: {total_plddt:.4f}')
        total_loss_x = 0
        total_loss_edge = 0
        total_vq = 0
        #total_plddt = 0
    torch.save(encoder.state_dict(), encoder_save)
    torch.save(decoder.state_dict(), decoder_save)