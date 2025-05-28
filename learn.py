from foldtree2 import foldtree2_ecddcd as ft2
from src.losses import fafe as fafe
from src import pdbgraph


from matplotlib import pyplot as plt
import numpy as np
import tqdm
import numpy as np
import glob
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pickle
import pandas as pd
import os
import time
import tqdm
import sys



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
import argparse

AVAIL_GPUS = min(1, torch.cuda.device_count())
datadir = '../../datasets/foldtree2/'

converter = pdbgraph.PDB2PyG( aapropcsv = 'config/aaindex1.csv' )
# Setting the seed for everything
torch.manual_seed(0)
np.random.seed(0)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add argparse for CLI configuration
parser = argparse.ArgumentParser(description='Train model with configurable parameters (scaling_experiment style)')
parser.add_argument('--dataset', '-d', type=str, default='structs_traininffttest.h5',
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
					help='Number of embeddings for the encoder (default: 50)')
parser.add_argument('--embedding-dim', type=int, default=20,
					help='Embedding dimension for the encoder (default: 20)')

# Add CLI arguments for overwrite, geometry, fapeloss, lddtloss, concat_positions, transformer
parser.add_argument('--overwrite' ,  action='store_true', help='Overwrite saved model if exists, otherwise continue training')
parser.add_argument('--geometry',  action='store_true', help='Train the model with geometry')
parser.add_argument('--fapeloss' ,   action='store_true', help='Train the model with FAPE loss')
parser.add_argument('--lddtloss',  action='store_true', help='Train the model with LDDT loss')
parser.add_argument('--concat-positions', action='store_true', help='Train the model with positional encoding')
parser.add_argument('--transformer' , action='store_true', help='Train the model with transformer decoder')
parser.add_argument('--output-foldx', action='store_true', help='Train the model with foldx output')

#print an overview of the arguments and example command if the user runs the script with -h
if len(sys.argv)==1:
	print('No arguments provided. Use -h for help.')
	print('Example command: python learn.py -d structs_training_godnodemk5.h5 -o ./models/ -lr 0.0001 -e 800 -bs 20 --geometry --fapeloss --lddtloss --concat-positions --transformer')
	print('Available arguments:')
	parser.print_help()
	sys.exit(0)

args = parser.parse_args()

# Set defaults for these options if not provided
overwrite = args.overwrite
geometry = args.geometry
fapeloss = args.fapeloss
lddtloss = args.lddtloss
concat_positions = args.concat_positions
transformer = args.transformer
output_foldx = args.output_foldx
# If transformer is enabled, force concat_positions to True
if transformer:
	concat_positions = True


# Use args for configuration
modeldir = args.output_dir
os.makedirs(modeldir, exist_ok=True)
modelname = args.model_name
batch_size = args.batch_size
lr = args.learning_rate
num_epochs = args.epochs
num_embeddings = args.num_embeddings
embedding_dim = args.embedding_dim

# Set device
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training loop
#load model if it exists

fftin = True
#apply weight initialization
applyinit_init = True
#clip gradients
clip_grad = True
#denoiser
denoise = True
#EMA for VQ
ema = False

edgeweight = .01
xweight = .01
vqweight = .0001
foldxweight = .001
fapeweight = .001
angleweight = .001
lddt_weight = .1
dist_weight = .01
err_eps = 1e-2

# Use args.dataset for dataset path
struct_dat = pdbgraph.StructureDataset(args.dataset)
# Create a DataLoader for training
train_loader = DataLoader(struct_dat, batch_size=batch_size, shuffle=True , worker_init_fn = np.random.seed(0) , num_workers=6)
for data_sample in train_loader:
	print('data sample:', data_sample)
	break
# Load a sample from the dataset
data_sample = next(train_loader)
ndim = data_sample['res'].x.shape[1] 
if fftin == True:
	ndim += data_sample['fourier1dr'].x.shape[1] + data_sample['fourier1di'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]

#model name
modelname = args.model_name

if os.path.exists(modeldir + modelname+'.pkl') and  overwrite == False:
	with open( modeldir +modelname + '.pkl', 'rb') as f:
		encoder, decoder = pickle.load(f)
	if args.geometry == True:
		decoder, geodecoder1, geodecoderSE3 = decoder
else:
	encoder_layers = 2
	encoder = ft2.mk1_Encoder(in_channels=ndim, hidden_channels=[args.hidden_size] * encoder_layers ,
							out_channels= embedding_dim , 
							metadata=  { 'edge_types': [     ('res','contactPoints', 'res') , ('res','hbond', 'res')  ] } , #, ('res','hbond', 'res') ,  ('res','backbone', 'res') ] }, 
							num_embeddings=num_embeddings, commitment_cost=.9 , edge_dim = 1 ,
							encoder_hidden=args.hidden_size , EMA = ema , nheads = 5 , dropout_p = 0.005 ,
								reset_codes= False , flavor = 'transformer' , fftin= fftin)
	
	decoder_layers = 3
	decoder = ft2.HeteroGAE_Decoder(in_channels = {'res':encoder.out_channels  , 'godnode4decoder':ndim_godnode ,
													'foldx':23 } , 
								hidden_channels={
												#( 'res','window','res'):[ args.hidden_size ] * decoder_layers  , 
												( 'res','backbone','res'):[ args.hidden_size] * decoder_layers  ,
												( 'res','backbonerev','res'):[ args.hidden_size ] * decoder_layers  ,
												('res' ,'informs','godnode4decoder' ):[ args.hidden_size] * decoder_layers ,
												},
								layers = decoder_layers ,
								metadata=converter.metadata , 
								amino_mapper = converter.aaindex ,
								concat_positions = concat_positions ,
								flavor = 'mfconv' ,
								output_foldx = output_foldx ,
								Xdecoder_hidden= [args.hidden_size, args.hidden_size//2 , max(1,args.hidden_size//5)  ] ,
								PINNdecoder_hidden = [ max(1,args.hidden_size//2) , max(1,args.hidden_size//4), max(1,args.hidden_size//5)] ,
								AAdecoder_hidden = [ args.hidden_size , args.hidden_size//2 , args.hidden_size//2 ] ,
								contactdecoder_hidden = [ max(1,args.hidden_size//2) , max(1,args.hidden_size//2)  ] ,
								nheads = 1, 
								dropout = 0.005  ,
								residual = False,
								normalize=True,
								contact_mlp=True
								)
	
	"""
	if args.geometry:
		geodecoder1 = CoordinateFreeTransformer(
			node_feature_dim: embedding_dim,
			hidden_dim: int = args.hidden_size, args.hidden_size//2 ,
			num_layers: int = 4,
			heads: int = 10,
			dropout: float = 0.05)
		
		geodecoderSE3 = SE3InvariantTransformer(
			node_feature_dim: embedding_dim,
			hidden_dim: int = args.hidden_size, args.hidden_size//2 ,
			num_layers: int = 4,
			heads: int = 10,
			num_degrees: int = 2,
			dropout: float = 0.05,
		)
	"""
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


for epoch in range(num_epochs):
	for data in tqdm.tqdm(train_loader):
		data = data.to(device)
		if init == False:
			with torch.no_grad():  # Initialize lazy modules.
				z,vqloss = encoder.forward(data )
				data['res'].x = z
				recon_x , edge_probs , zgodnode , foldxout = decoder(  data , None  ) 
				init = True
				#nparameters
				print('nparams:' ,  sum(p.numel() for p in encoder.parameters() ) + sum(p.numel() for p in decoder.parameters() ) )
				continue
		
		optimizer.zero_grad()
		z,vqloss = encoder.forward(data )
		data['res'].x = z
		edgeloss , distloss = ft2.recon_loss(  data , data.edge_index_dict[('res', 'contactPoints', 'res')] , decoder  , plddt= False , offdiag = False )
		recon_x , edge_probs , zgodnode , foldxout  = decoder(  data , None )
		

		angleloss = torch.tensor(0.0).to(device)
		fapeloss = torch.tensor(0.0).to(device)
		lddtloss = torch.tensor(0.0).to(device)

		"""
		#compute geometry losses
		if geometry == True:
			#compute the geometry loss
			r1,t1, a1 = geodecoder1(data)
			#transform r1 and t1 to coordinates
			data['coord_reconstruction'] = 
			r2,t2, a2 = geodecoderSE3(data)


			#modulate angles row-wise with plddt
			angleloss = F.smooth_l1_loss( angles , data.x_dict['bondangles'] , reduction='none' )
			fapeloss = torch.tensor(0.0).to(device)
			lddt_loss = torch.tensor(0.0).to(device)
			angleloss += F.smooth_l1_loss( angles2 , data.x_dict['bondangles'] , reduction='none' )
			if fapeloss == True:
				#reshape the data into batched form
				batch = data['t_true'].batch
				t_true = data['t_true'].x
				R_true = data['R_true'].x
				#Compute the FAPE loss
				fapeloss += ft2.fape_loss(true_R = R_true,
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
		"""

		#compute the amino acide reconstruction loss
		xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
		if decoder.output_foldx == True:
			data['Foldx'].x = data['Foldx'].x.view(-1, 23)
			data['Foldx'].x  = decoder.bn_foldx(data['Foldx'].x)
			foldxout = foldxout.view(data['Foldx'].x.shape)
			foldxloss = F.smooth_l1_loss( foldxout , data['Foldx'].x )
		else:
			foldxloss = torch.tensor(0.0)
		
		
		#plddtloss = x_reconstruction_loss(data['plddt'].x, recon_plddt)
		loss = xweight*xloss + edgeweight*edgeloss + vqweight*vqloss 
		loss += foldxloss*foldxweight + fapeweight*fapeloss + angleweight*angleloss+ lddt_weight*lddtloss 
		loss.backward()
		
		if clip_grad:
			torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
			torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
		
		
		optimizer.step()		
		total_loss_edge += edgeloss.item()
		total_loss_x += xloss.item()
		total_vq += vqloss.item()		
		total_angleloss += angleloss.item()
		total_lddtloss += lddtloss.item()
		total_fapeloss += fapeloss.item()
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
		with open(os.path.join(modeldir, modelname + '.pkl'), 'wb') as f:
			print( encoder , decoder )
			pickle.dump( (encoder, decoder) , f)
	trainlen = len(train_loader)

	print(f'Epoch {epoch}, AALoss: {total_loss_x/trainlen:.4f}, Edge Loss: {total_loss_edge/trainlen:.4f}, vq Loss: {total_vq/trainlen:.4f} , foldx Loss: {total_foldx/trainlen:.4f}' ) 
	print(f'fapeloss: {total_fapeloss/trainlen:.4f} , angleloss: {total_angleloss/trainlen:4f} , lddtloss: {total_lddtloss/trainlen:4f} , distloss: {total_distloss/trainlen:4f}' )
	print( f'total_loss: {(total_loss_x + total_loss_edge + total_vq + total_foldx)/trainlen:.4f}' )
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

with open(os.path.join(modeldir, modelname + '.pkl'), 'wb') as f:
	pickle.dump( (encoder, decoder) , f)
