
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
import time
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
AVAIL_GPUS = min(1, torch.cuda.device_count())
datadir = '../../datasets/foldtree2/'
converter = pdbgraph.PDB2PyG()
# Setting the seed for everything
torch.manual_seed(0)
np.random.seed(0)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#set the directory for the datasets
datadir = '../../datasets/'
modeldir = './models/'

# Training loop
#load model if it exists

#overwrite saved model
overwrite = True
#set to true to train the model with geometry
geometry = True
#set to true to train the model with fape loss
fapeloss = True
#set to true to train the model with lddt loss
lddtloss = False
#set to true to train the model with positional encoding
concat_positions = True
#set to true to train the model with transformer
transformer = False
if transformer == True:	
	concat_positions = True

#apply weight initialization
applyinit_init = True
#clip gradients
clip_grad = True
#denoiser
denoise = True
#EMA for VQ
ema = True

edgeweight = .01
xweight = .01
vqweight = .0001
foldxweight = .001
fapeweight = .01
angleweight = .01
lddt_weight = .1
dist_weight = .01
err_eps = 1e-2
lr = 0.0001
batch_size = 20
num_embeddings = 50
embedding_dim = 20

struct_dat = pdbgraph.StructureDataset('structs_training_godnodemk5.h5')
# Create a DataLoader for training
train_loader = DataLoader(struct_dat, batch_size=batch_size, shuffle=True , worker_init_fn = np.random.seed(0) , num_workers=6)
# Load a sample from the dataset
data_sample = next(iter(train_loader))
ndim = data_sample['res'].x.shape[1] 
ndim_godnode = data_sample['godnode'].x.shape[1]  

#model name
modelname = 'newmodelmk6tanh'

if os.path.exists(modeldir + modelname+'.pkl') and  overwrite == False:
	with open( modeldir +modelname + '.pkl', 'rb') as f:
		encoder, decoder = pickle.load(f)
else:
	encoder_layers = 2
	encoder = ft2.mk1_Encoder(in_channels=ndim, hidden_channels=[ 200 ] *  encoder_layers ,
							out_channels= embedding_dim , 
							metadata=  { 'edge_types': [     ('res','contactPoints', 'res') , ('res','backbone', 'res') ,('res','backbonerev', 'res') ] } , #, ('res','hbond', 'res') ,  ('res','backbone', 'res') ] }, 
							num_embeddings=num_embeddings, commitment_cost=.9 , edge_dim = 1 ,
							encoder_hidden=300 , EMA = ema , nheads = 10 , dropout_p = 0.005 ,
								reset_codes= False , flavor = 'transformer' )
	if transformer == True:
		decoder_layers = 2
		decoder = ft2.Transformer_Decoder(in_channels = {'res':encoder.out_channels  , 'godnode4decoder':ndim_godnode ,
														'foldx':23 } , 
									layers = decoder_layers ,
									metadata=converter.metadata , 
									amino_mapper = converter.aaindex ,
									concat_positions = concat_positions ,
									output_foldx = True ,
									normalize= True ,
									geometry= geometry ,
									denoise = denoise ,
									Xdecoder_hidden= 50 ,
									PINNdecoder_hidden = [ 10 , 10, 10] ,
									contactdecoder_hidden = [ 20 , 10] ,
									nheads = 4, 
									dropout = 0.001  ,
									AAdecoder_hidden = [20 , 10 , 10]  ,
									)    
	else:	
		decoder_layers = 5
		decoder = ft2.HeteroGAE_Decoder(in_channels = {'res':encoder.out_channels  , 'godnode4decoder':ndim_godnode ,
														'foldx':23 } , 
									hidden_channels={
													#( 'res','window','res'):[ 20 ] * decoder_layers  , 
													( 'res','backbone','res'):[ 200] * decoder_layers  ,
													( 'res','backbonerev','res'):[ 200 ] * decoder_layers  ,
													('res' ,'informs','godnode4decoder' ):[ 200] * decoder_layers ,
													},
									layers = decoder_layers ,
									metadata=converter.metadata , 
									amino_mapper = converter.aaindex ,
									concat_positions = concat_positions ,
									flavor = 'sage' ,
									output_foldx = True ,
									geometry= geometry ,
									denoise = denoise ,
									Xdecoder_hidden= [500, 100 , 100  ] ,
									PINNdecoder_hidden = [ 50 , 50, 20] ,
									geodecoder_hidden = [30 , 30, 30 ] ,
									AAdecoder_hidden = [ 500 , 100 , 100 ] ,
									contactdecoder_hidden = [ 100 , 100 ] ,
									nheads = 10, 
									dropout = 0.005  ,
									residual = False,
									normalize=True,
									contact_mlp=True
									)
    
print('encoder', encoder)
print('decoder', decoder)

#save a file with timestamp modelname and parameters


def init_weights(m):
	#init the heteroconv weights
	if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d) :
		torch.nn.init.xavier_uniform_(m.weight)
	#if m.bias is not None:
	#	torch.nn.init.zeros_(m.bias)
			
if applyinit_init == True:
	encoder.apply(init_weights)
	decoder.apply(init_weights)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

#put encoder and decoder on the device
encoder = encoder.to(device)
decoder = decoder.to(device)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr  )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

encoder.train()
decoder.train()
xlosses = []
edgelosses = []
vqlosses = []
foldxlosses = []
fapelosses = []


with open( modeldir + modelname + 'run.txt', 'w') as f:
	#write date and time of run
	f.write( 'date: ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
	f.write( 'encoder: ' + str(encoder) + '\n')
	f.write( 'decoder: ' + str(decoder) + '\n')
	f.write( 'geometry: ' + str(geometry) + '\n')
	f.write( 'fapeloss: ' + str(fapeloss) + '\n')
	f.write( 'lddtloss: ' + str(lddtloss) + '\n')
	f.write( 'concat_positions: ' + str(concat_positions) + '\n')
	f.write( 'transformer: ' + str(transformer) + '\n')
	f.write( 'modelname: ' + modelname + '\n')
	#write weights for losses
	f.write( 'edgeweight: ' + str(edgeweight) + '\n')
	f.write( 'xweight: ' + str(xweight) + '\n')
	f.write( 'vqweight: ' + str(vqweight) + '\n')
	f.write( 'foldxweight: ' + str(foldxweight) + '\n')
	f.write( 'fapeweight: ' + str(fapeweight) + '\n')
	f.write( 'angleweight: ' + str(angleweight) + '\n')
	f.write( 'lddt_weight: ' + str(lddt_weight) + '\n')
	f.write( 'dist_weight: ' + str(dist_weight) + '\n')

#log parameters w tensorboard
writer.add_text('Parameters', 'encoder: ' + str(encoder) , 0)
writer.add_text('Parameters', 'decoder: ' + str(decoder) , 0)
writer.add_text('Parameters', 'geometry: ' + str(geometry) , 0)
writer.add_text('Parameters', 'fapeloss: ' + str(fapeloss) , 0)
writer.add_text('Parameters', 'lddtloss: ' + str(lddtloss) , 0)
writer.add_text('Parameters', 'concat_positions: ' + str(concat_positions) , 0)
writer.add_text('Parameters', 'transformer: ' + str(transformer) , 0)
writer.add_text('Parameters', 'modelname: ' + modelname , 0)
writer.add_text('Parameters', 'edgeweight: ' + str(edgeweight) , 0)
writer.add_text('Parameters', 'xweight: ' + str(xweight) , 0)
writer.add_text('Parameters', 'vqweight: ' + str(vqweight) , 0)
writer.add_text('Parameters', 'foldxweight: ' + str(foldxweight) , 0)
writer.add_text('Parameters', 'fapeweight: ' + str(fapeweight) , 0)
writer.add_text('Parameters', 'angleweight: ' + str(angleweight) , 0)
writer.add_text('Parameters', 'lddt_weight: ' + str(lddt_weight) , 0)
writer.add_text('Parameters', 'dist_weight: ' + str(dist_weight) , 0)


total_loss_x= 0
total_loss_edge = 0
total_vq=0
total_kl = 0
total_foldx=0
total_fapeloss = 0
total_lddtloss = 0
total_angleloss = 0
total_distloss = 0

init = False



def analyze_gradient_norms(model, top_k=3):
    """
    Analyzes gradients in the given model and returns the top_k layers with
    highest and lowest gradient norms.

    Parameters:
        model (torch.nn.Module): The neural network model after backpropagation.
        top_k (int): Number of layers to return for highest and lowest gradients.

    Returns:
        dict: {'highest': [(layer_name, grad_norm)], 'lowest': [(layer_name, grad_norm)]}
    """
    grad_norms = []

    # Collect gradient norms
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))

    # Sort by gradient norms
    grad_norms.sort(key=lambda x: x[1])

    # Get lowest and highest
    lowest = grad_norms[:top_k]
    highest = grad_norms[-top_k:][::-1]

    return {'highest': highest, 'lowest': lowest}


for epoch in range(800):
	for data in tqdm.tqdm(train_loader):
		data = data.to(device)
		if init == False:
			with torch.no_grad():  # Initialize lazy modules.

				z,vqloss = encoder.forward(data )
				data['res'].x = z
				recon_x , edge_probs , zgodnode , foldxout, r , t , angles , r2,t2, angles2 = decoder(  data , None  ) 
				init = True
				#nparameters
				print('nparams:' ,  sum(p.numel() for p in encoder.parameters() ) + sum(p.numel() for p in decoder.parameters() ) )

				continue
		
		optimizer.zero_grad()
		z,vqloss = encoder.forward(data )
		data['res'].x = z
		edgeloss , distloss = ft2.recon_loss(  data , data.edge_index_dict[('res', 'contactPoints', 'res')] , decoder  , plddt= False , offdiag = False )
		recon_x , edge_probs , zgodnode , foldxout , r , t , angles , r2,t2,angles2 = decoder(  data , None )
		
		#compute geometry losses
		if geometry == True:
			#modulate angles row-wise with plddt
			angleloss = F.smooth_l1_loss( angles , data.x_dict['bondangles'] , reduction='none' )
			fploss = torch.tensor(0.0).to(device)
			lddt_loss = torch.tensor(0.0).to(device)
			if denoise == True:
				angleloss += F.smooth_l1_loss( angles2 , data.x_dict['bondangles'] , reduction='none' )
				if fapeloss == True:
					#reshape the data into batched form
					batch = data['t_true'].batch
					t_true = data['t_true'].x
					R_true = data['R_true'].x
					#Compute the FAPE loss
					fploss += ft2.fape_loss(true_R = R_true,
							true_t = t_true, 
							pred_R = r2, 
							pred_t = t2, 
							batch = batch, 
							d_clamp = 10.0,
							eps=1e-8,
							plddt = None ,
							soft = False )
				if lddtloss == True:
					lddt_loss += ft2.lddt_loss(
							coord_true= data['coords'].x,
							pred_R = r, 
							pred_t = t, 
							batch = batch
							)
			angleloss = angleloss.mean()
		else:
			fploss = torch.tensor(0.0).to(device)
			lddt_loss = torch.tensor(0.0).to(device)
			angleloss = torch.tensor(0.0).to(device)
	
		#compute the amino acide reconstruction loss
		xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
		if decoder.output_foldx == True:
			data['Foldx'].x = data['Foldx'].x.view(-1, 23)
			data['Foldx'].x  = decoder.bn_foldx(data['Foldx'].x)
			foldxout = foldxout.view(data['Foldx'].x.shape)
			foldxloss = F.smooth_l1_loss( foldxout , data['Foldx'].x )
		else:
			foldxloss = torch.tensor(0.0)
		
		for l in [ xloss , edgeloss , vqloss , foldxloss , fploss, angleloss , lddt_loss  ]:
			if torch.isnan(l).any():
				l = 0
		
		#plddtloss = x_reconstruction_loss(data['plddt'].x, recon_plddt)
		loss = xweight*xloss + edgeweight*edgeloss + vqweight*vqloss 
		loss += foldxloss*foldxweight + fapeweight*fploss + angleweight*angleloss+ lddt_weight*lddt_loss 
		loss.backward()
		
		if clip_grad:
			torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
			torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
		
		
		optimizer.step()
		total_loss_edge += edgeloss.item()
		total_loss_x += xloss.item()
		total_vq += vqloss.item()		
		total_angleloss += angleloss.item()
		total_lddtloss += lddt_loss.item()
		total_fapeloss += fploss.item()
		total_distloss += distloss.item()

		if decoder.output_foldx == True:
			total_foldx += foldxloss.item()
		else:
			total_foldx = 0

	scheduler.step(total_loss_x)
	
	if total_foldx < err_eps:
		foldxweight = 0
	else:
		foldxweight = 0.01

	#save the best model
	if epoch % 10 == 0 and epoch > 0:
		#save model
		print( 'saving model')
		with open( modeldir + modelname + '.pkl', 'wb') as f:
			print( encoder , decoder )
			pickle.dump( (encoder, decoder) , f)
	print(f'Epoch {epoch}, AALoss: {total_loss_x*batch_size:.4f}, Edge Loss: {total_loss_edge*batch_size:.4f}, vq Loss: {total_vq*batch_size:.4f} , foldx Loss: {total_foldx*batch_size:.4f}' ) 
	print(f'fapeloss: {total_fapeloss*batch_size:.4f} , angleloss: {total_angleloss*batch_size:4f} , lddtloss: {total_lddtloss/batch_size:4f} , distloss: {total_distloss*batch_size:4f}' )
	print( 'grad encoder:' ,  analyze_gradient_norms(encoder) )
	print('grad decoder:' ,  analyze_gradient_norms(decoder) )

	writer.add_scalar('Loss/AA', total_loss_x, epoch)
	writer.add_scalar('Loss/Edge', total_loss_edge, epoch)
	writer.add_scalar('Loss/VQ', total_vq, epoch)
	writer.add_scalar('Loss/Foldx', total_foldx, epoch)
	writer.add_scalar('Loss/Fape', total_fapeloss, epoch)
	writer.add_scalar('Loss/Angle', total_angleloss, epoch)
	writer.add_scalar('Loss/LDDT', total_lddtloss, epoch)
	writer.add_scalar('Loss/Dist', total_distloss, epoch)
	#log learning rate
	writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
	total_loss_x = 0
	total_loss_edge = 0
	total_vq = 0
	total_foldx = 0
	total_fapeloss = 0
	total_angleloss = 0
	total_lddtloss = 0

with open( modeldir + modelname+'.pkl', 'wb') as f:
	pickle.dump( (encoder, decoder) , f)
