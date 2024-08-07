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

encoder_save = 'encoder_mk3_aa_EMA_40_lowcost_small10_transformer'
decoder_save = 'decoder_mk3_aa_EMA_40_lowcost_small10_transformer'

model_dir = 'models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

overwrite =  True
train_loop = True
variational = False
betafactor = 2

struct_dat = ft2.StructureDataset('structs_training.h5')

# Create a DataLoader for training
total_loss_x = 0
total_loss_edge = 0
total_vqe=0
total_vqq=0
total_vq = 0
total_kl = 0
total_angle = 0
total_plddt=0


contact_ratio = 0.1
calc_ratio = False

# Training loop
ndim = struct_dat[0]['res'].x.shape[1]
print( struct_dat[0] )

#load model if it exists
if variational == True:
    encoder = ft2.HeteroGAE_variational_Encoder(in_channels=ndim, hidden_channels=[ 300 ]*4 , out_channels=100, metadata=converter.metadata , num_embeddings=64, 
                                commitment_cost=.8 , encoder_hidden=50 , EMA = True , reset_codes= False )

else:
    #add positional encoder channels to input
    encoder = ft2.HeteroGAE_Encoder(in_channels=ndim, hidden_channels=[ 100 ]*4 , out_channels=20, metadata=converter.metadata , num_embeddings=50, 
                                commitment_cost=.8 , encoder_hidden=100 , EMA = True , reset_codes= False )



decoder = ft2.HeteroGAE_Decoder(encoder_out_channels = encoder.out_channels , 
                            hidden_channels={ ( 'res','backbone','res'):[ 1000 ] * 5  } , 
                            out_channels_hidden= 100 , metadata=converter.metadata , amino_mapper = converter.aaindex , Xdecoder_hidden=100 , predangles=True) 


if os.path.exists(encoder_save) and overwrite == False:
    encoder.load_state_dict(torch.load(model_dir+encoder_save ))
    print('Loaded encoder')
if os.path.exists(decoder_save) and overwrite == False:
    decoder.load_state_dict(torch.load(model_dir+decoder_save  ))
    print('Loaded decoder')


#create a training loop for the GAE model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')

device = torch.device("cuda:1")
print(device)

#put encoder and decoder on the device
encoder = encoder.to(device)
decoder = decoder.to(device)
sharpen = False

if train_loop == True:
    train_loader = DataLoader(struct_dat, batch_size=40, shuffle=True)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    encoder.train()
    decoder.train()
    
    all_losses = []
    edgeweight = 1
    xweight = 1
    vqweight = 1
    plddtweight = 1
    angle_weight = 1

    for epoch in range(1000):
        count = 0
        for data in tqdm.tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            if variational == True:
                z, mu, logvar, vqloss , eloss, qloss = encoder.forward(data['res'].x, data['AA'].x , data.edge_index_dict)
            else:
                z,vqloss,eloss,qloss = encoder.forward(data['res'].x, data['AA'].x , data.edge_index_dict)
            #z = torch.cat([z, data['positions'].x], dim=1)
            #add positional encoding to give to the decoder
            edgeloss = ft2.recon_loss(z , data.edge_index_dict[( 'res','contactPoints','res')]
                                  , data.edge_index_dict[( 'res','backbone','res')], decoder)
            
            if calc_ratio == True:
                contact_ratio = data.edge_index_dict[( 'res','contactPoints','res')].shape[1] / data.edge_index_dict[( 'res','backbone','res')].shape[1]**2
            else:
                contact_ratio = contact_ratio
            decode_out= decoder(z , data.edge_index_dict[( 'res','contactPoints','res')] , data.edge_index_dict , poslossmod = -np.log(contact_ratio) , neglossmod= -np.log(1-contact_ratio) )     
            xloss = ft2.aa_reconstruction_loss(data['AA'].x, decode_out[0])
            
            if decoder.predangles == True:
                angles = decode_out[2]
                #l1 loss on angles
                angle_loss = torch.nn.functional.smooth_l1_loss(angles,data['bondangles'].x)
            else:
                angle_loss = 0
            
            if sharpen == True:
                vqloss = eloss + qloss
                vqweight = 2
                edgeweight = .5
                xweight = .5
                angle_weight = .5
            
            loss = xweight*xloss + edgeweight*edgeloss + vqweight*vqloss + angle_weight*angle_loss

            if variational == True:
                # KL Divergence loss
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss += kl_loss * betafactor
        
            loss.backward()
            optimizer.step()
            total_loss_edge += edgeloss.item()
            total_loss_x += xloss.item()
            total_vqe += eloss.item()
            total_vqq += qloss.item()
            total_vq += vqloss.item()
            total_angle += angle_loss.item()
            count += 1
        
        if total_loss_edge/count <  1:
            sharpen = True
            print('Sharpening quantization')
        else:
            sharpen = False

        all_loss = total_loss_x + total_loss_edge + total_vq 
        all_losses.append(all_loss)
        if len(all_losses) > 10:    
            #remove first loss
            all_losses.pop(0)
            #if loss isnt decreasing, lower the learning rate
            #compute average delta of loss between epoch over the last 10 epochs
            delta = np.mean([ all_losses[i] - all_losses[i+1] for i in range( len(all_losses) - 1 ) ] )
            if delta < 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.1
                    print(f'Lowering learning rate to {g["lr"]}')    
                    all_losses = [] 
        if epoch % 5 == 0:
            #save model
            torch.save(encoder.state_dict(), encoder_save)
            torch.save(decoder.state_dict(), decoder_save)
            with open(model_dir + encoder_save + '_model.pkl' , 'wb') as f:
                pickle.dump([encoder, decoder], f)

        print(f'Epoch {epoch}, AALoss: {total_loss_x/count:.4f}, Angle Loss: {total_angle/count:.4f} ,  Edge Loss: {total_loss_edge/count:.4f}, vq e Loss: {total_vqe/count:.4f}, vq q Loss: {total_vqq/count:.4f} , vq Loss: {total_vq/count:.4f}')
        total_loss_x = 0
        total_loss_edge = 0
        total_vq = 0
        total_vqe = 0
        total_vqq = 0
        total_angle = 0
        #total_plddt = 0
    torch.save(encoder.state_dict(), encoder_save)
    torch.save(decoder.state_dict(), decoder_save)
