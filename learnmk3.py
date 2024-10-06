import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pickle
import os
import numpy as np
import tqdm
import datetime

import foldtree2_ecddcd as ft2

converter = ft2.PDB2PyG()

encoder_save = 'encoder_lowcost_final'
decoder_save = 'decoder_lowcost_final'

model_dir = 'models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

overwrite =  False
train_loop = True
variational = False
betafactor = 2

struct_dat = ft2.StructureDataset('structs_training_godnodemk3.h5')

# Create a DataLoader for training
total_loss_x = 0
total_loss_edge = 0
total_vqe=0
total_vqq=0
total_vq = 0
total_kl = 0
total_angle = 0
total_plddt=0


contact_ratio = 1
calc_ratio = False

separated = True

# Training loop
ndim = struct_dat[0]['res'].x.shape[1]

batch_size = 40
print( struct_dat[0] )
#load model if it exists

encoder = ft2.HeteroGAE_Encoder(in_channels={'res':ndim}, hidden_channels=[400, 400, 400 ] , 
                        out_channels=10, metadata=converter.metadata , num_embeddings=64, 
                        commitment_cost= .8 , encoder_hidden= 200  , nheads = 4 , average = False
                        , reset_codes= False , dropout_p=0.001 , separated = separated , flavor = 'mfconv' )

decoder = ft2.HeteroGAE_Decoder(encoder_out_channels = encoder.out_channels , 
                            hidden_channels={ ( 'res','backbone','res'):[  400 ] * 5  }   , 
                            metadata=converter.metadata , 
                            amino_mapper = converter.aaindex 
                            , Xdecoder_hidden= 500 , nheads = 4 , dropout = 0.001  ,
                              AAdecoder_hidden = [400,400,400] , flavor = 'SAGE' )    

#log all hyperparameters and batch size
with open(model_dir + encoder_save + '_params.txt' , 'w') as f:
    #write time of training
    time = datetime.datetime.now()
    f.write(f'Time: {time}'  )
    f.write(f'Encoder: {encoder}\n')
    f.write(f'Decoder: {decoder}\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Variational: {variational}\n')
    f.write(f'Beta factor: {betafactor}\n')
    f.write(f'Contact ratio: {contact_ratio}\n')
    f.write(f'Calculate ratio: {calc_ratio}\n')
    f.write(f'Separated: {separated}\n')
    f.write(f'Overwrite: {overwrite}\n')

if os.path.exists(model_dir+encoder_save) and overwrite == False:
    encoder.load_state_dict(torch.load(model_dir+encoder_save ))
    print('Loaded encoder')
if os.path.exists(model_dir+decoder_save) and overwrite == False:
    decoder.load_state_dict(torch.load(model_dir+decoder_save  ))
    print('Loaded decoder')


print('encoder',encoder)

print('decoder' , decoder)


#create a training loop for the GAE model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')

device = torch.device("cuda:1")
#device = torch.device("cpu")
print(device)

#put encoder and decoder on the device
encoder = encoder.to(device)
decoder = decoder.to(device)
sharpen = False

if train_loop == True:
    train_loader = DataLoader(struct_dat, batch_size=batch_size, shuffle=True)
    
    all_losses = []
    edgeweight = 1
    xweight = 1
    vqweight = 1
    plddtweight = 1

    #optimizer = torch.optim.Adagrad( list(encoder.parameters()) + list(decoder.parameters()) , lr = 0.001)
    optimizer = torch.optim.Adam( list(encoder.parameters()) + list(decoder.parameters()) , lr = 0.0005)
    #optimizer = torch.optim.RMSprop( list(encoder.parameters()) + list(decoder.parameters()) , lr = 0.001)
    encoder.train()
    decoder.train()

    for epoch in range(1000):
        count = 0
        for data in tqdm.tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
        
            z,qloss = encoder.forward(data.x_dict, data.edge_index_dict)
            edgeloss = ft2.recon_loss(z , data.edge_index_dict[( 'res','contactPoints','res')]
                                  , data.edge_index_dict[( 'res','backbone','res')], decoder)
            
            decode_out = decoder(z , data.edge_index_dict[( 'res','contactPoints','res')] , data.edge_index_dict , poslossmod = 1 , neglossmod= 1 )
            xloss = ft2.aa_reconstruction_loss(data['AA'].x, decode_out[0])           
            loss = xweight*xloss + edgeweight*edgeloss + vqweight*qloss 

            if variational == True:
                # KL Divergence loss
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss += kl_loss * betafactor
        
            loss.backward()
            optimizer.step()

            total_loss_edge += edgeloss.item()
            total_loss_x += xloss.item()
            total_vqq += qloss.item()
            
            
            count += 1
       
        all_loss = total_loss_x + total_loss_edge + total_vq 
        all_losses.append(all_loss)

        if epoch > 50 and epoch % 50 == 0:
            #reduce learning rate if loss isnt decreasing
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
                print(f'Lowering learning rate to {g["lr"]}')
            #change the loss weight of vq to only sharpen
            encoder.vector_quantizer.entropyweight= 0
            encoder.vector_quantizer.diversityweight = 0
            encoder.vector_quantizer.klweight = 0
        
        if epoch == 0:
            xlossmin = total_loss_x

        if total_loss_x < xlossmin:
            #save model
            torch.save(encoder.state_dict(), model_dir+encoder_save)
            torch.save(decoder.state_dict(), model_dir+decoder_save)
            with open(model_dir + encoder_save + 'proto_model.pkl' , 'wb') as f:
                pickle.dump([encoder, decoder], f)

        print(f'Epoch {epoch}, AALoss: {total_loss_x:.4f},  Edge Loss: {total_loss_edge:.4f} , vq Loss: {total_vqq/count:.4f}')
        total_loss_x = 0
        total_loss_edge = 0
        total_vq = 0
        total_vqe = 0
        total_vqq = 0
        total_angle = 0
        #total_plddt = 0
    torch.save(encoder.state_dict(), encoder_save)
    torch.save(decoder.state_dict(), decoder_save)
