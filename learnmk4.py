
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
writer = SummaryWriter()
import numpy as np
AVAIL_GPUS = min(1, torch.cuda.device_count())
#download an example pdb file
filename = './1eei (1).pdb'
url = 'https://files.rcsb.org/download/1EEI.pdb'
#filename = wget.download(url)
datadir = '../../datasets/foldtree2/'
filename = './1eei.pdb'
converter = pdbgraph.PDB2PyG()
res  = converter.create_features(filename, distance = 10, verbose = False )
angles, contact_points, springmat , hbond_mat, backbone , backbone_rev , positional_encoding , plddt , aa , bondangles , foldxvals, coords ,window , windowrev = res
# Setting the seed for everything
torch.manual_seed(0)
np.random.seed(0)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#data_sample =converter.struct2pyg( pdbfiles[0] , verbose=False)
#print(data_sample)
#ndim = data_sample['res'].x.shape[1]
#ndim_godnode = data_sample['godnode'].x.shape[1]
datadir = '../../datasets/'
pdbfiles = glob.glob(datadir +'structs/*.pdb')
data_sample =converter.struct2pyg( pdbfiles[0], foldxdir='./foldx/',  verbose=False)
ndim = data_sample['res'].x.shape[1] 
ndim_godnode = data_sample['godnode'].x.shape[1]  

# Training loop
#load model if it exists
encoder_layers = 2
decoder_layers = 5

overwrite = True
fapeloss = False

encoder_save = 'contactmlp_hbond_nogeo_noema'
decoder_save = 'contactmlp_hbond_nogeo_noema'
modelname = 'contactmlp_hbond_nogeo_noema'

if os.path.exists(encoder_save) and os.path.exists(decoder_save) and overwrite == False:
	with open( modelname + '.pkl', 'rb') as f:
		encoder, decoder = pickle.load(f)
else:
	encoder = ft2.mk1_Encoder(in_channels=ndim, hidden_channels=[ 100 ]*encoder_layers ,
							out_channels=20, metadata=converter.metadata , 
							num_embeddings=40, commitment_cost=.9 , edge_dim = 1 ,
							encoder_hidden=100 , EMA = True , nheads = 8 , dropout_p = 0.001 ,
								reset_codes= False , flavor = 'gat' )

	decoder = ft2.HeteroGAE_Decoder(in_channels = {'res':encoder.out_channels + 256 , 'godnode4decoder':ndim_godnode ,
													'foldx':23 } , 
								hidden_channels={
												('res' ,'informs','godnode4decoder' ):[  75 ] * decoder_layers ,
												#('godnode4decoder' ,'informs','res' ):[  75 ] * decoder_layers ,
												( 'res','backbone','res'):[ 75 ] * decoder_layers , 
												#('res' , 'backbonerev' , 'res'): [75] * decoder_layers ,
												},
								layers = decoder_layers ,
								metadata=converter.metadata , 
								amino_mapper = converter.aaindex ,
								flavor = 'sage' ,
								output_foldx = True ,
								contact_mlp = False ,
								denoise = fapeloss ,
								Xdecoder_hidden= 100 ,
								PINNdecoder_hidden = [100 , 50, 10] ,
								contactdecoder_hidden = [50 , 50 ] ,
								nheads = 4, dropout = 0.001  ,
								AAdecoder_hidden = [100 , 100 , 20]  )    


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)

#encoder.apply(init_weights)
#decoder.apply(init_weights)

#load mean and variance	and turn them into tensors	
mean = pd.read_csv('foldxmean.csv', index_col = 0)
variance = pd.read_csv('foldxvariance.csv', index_col = 0)
mean = torch.tensor(mean.values).float()
variance = torch.tensor(variance.values).float()


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')
print(device)
batch_size = 20

#put encoder and decoder on the device
encoder = encoder.to(device)
decoder = decoder.to(device)

struct_dat = pdbgraph.StructureDataset('structs_training_godnodemk4.h5')
err_eps = 1e-2

# Create a DataLoader for training

train_loader = DataLoader(struct_dat, batch_size=batch_size, shuffle=True , worker_init_fn = np.random.seed(0) , num_workers=6)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001  )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

encoder.train()
decoder.train()
xlosses = []
edgelosses = []
vqlosses = []
foldxlosses = []
fapelosses = []

edgeweight = .1
xweight = 1
vqweight = .1
foldxweight = .01
fapeweight = .1
angleweight = .1

total_loss_x= 0
total_loss_edge = 0
total_vq=0
total_kl = 0
total_foldx=0
total_fapeloss = 0
total_angleloss = 0
init = False
for epoch in range(800):
	if epoch > 500:
		vqweight = 1
	for data in tqdm.tqdm(train_loader):
		data = data.to(device)
		if init == False:
			with torch.no_grad():  # Initialize lazy modules.
				z,vqloss = encoder.forward(data)
				z = torch.cat( (z, data.x_dict['positions'] ) , dim = 1)
				data['res'].x = z
				recon_x , edge_probs , zgodnode , foldxout, r , t , angles = decoder(  data , None ) 
				init = True
				continue
		
		optimizer.zero_grad()
		#normalize the foldx values
		z,vqloss = encoder.forward(data ) 
		#add positional encoding to y
		z = torch.cat( (z, data.x_dict['positions'] ) , dim = 1)
		data['res'].x = z
		#change backbone to undirected
		edgeloss = ft2.recon_loss(  data , data.edge_index_dict[('res', 'contactPoints', 'res')] , decoder , distweight=False)
		recon_x , edge_probs , zgodnode , foldxout , r , t , angles = decoder(  data , None ) 
		xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
		if decoder.output_foldx == True:
			data['Foldx'].x = data['Foldx'].x.view(-1, 23)
			data['Foldx'].x  = decoder.bn_foldx(data['Foldx'].x)
			foldxout = foldxout.view(data['Foldx'].x.shape)
			foldxloss = F.smooth_l1_loss( foldxout , data['Foldx'].x )
		else:
			foldxloss = 0

		if fapeloss:
			#reshape the data into batched form
			batch = data['t_true'].batch
			t_true = data['t_true'].x
			R_true = data['R_true'].x
			#Compute the FAPE loss
			fploss = ft2.fape_loss(true_R = R_true,
					 true_t = t_true, 
					 pred_R = r, 
					 pred_t = t, 
					 batch = batch, 
					 d_clamp = 10.0,
					 eps=1e-8,
					 plddt = data['plddt'].x,
					 soft = False )
			angleloss = F.smooth_l1_loss( angles*data['plddt'].x , data.x_dict['bondangles']*data['plddt'].x )
		else:
			fploss = torch.tensor(0)
			angleloss = torch.tensor(0)
		
		for l in [ xloss , edgeloss , vqloss , foldxloss , fploss, angleloss]:
			if torch.isnan(l).any():
				l = 0
		
		#plddtloss = x_reconstruction_loss(data['plddt'].x, recon_plddt)
		loss = xweight*xloss + edgeweight*edgeloss + vqweight*vqloss + foldxloss*foldxweight + fapeweight*fploss + angleweight*angleloss
		loss.backward()
		torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=100.0)
		torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=100.0)
		optimizer.step()
		total_loss_edge += edgeloss.item()
		total_loss_x += xloss.item()
		total_vq += vqloss.item()		
		total_angleloss += angleloss.item()

		if decoder.output_foldx == True:
			total_foldx += foldxloss.item()
		else:
			total_foldx = 0
		if fapeloss:
			total_fapeloss += fploss.item()
		else:
			total_fapeloss = 0

	scheduler.step(total_loss_x)

	if total_loss_x < 1:
		xweight = 10
	if total_loss_x < err_eps:
		xweight = 0 
	else:
		xweight = 1
	
	if total_foldx < err_eps:
		foldxweight = 0
	else:
		foldxweight = .01
	
	if total_vq < 8:
		vqweight = 0
	else:
		vqweight = 1

	#save the best model
	if epoch % 10 == 0 :
		#save model
		print( 'saving model')
		with open( modelname + '.pkl', 'wb') as f:
			print( encoder , decoder )
			pickle.dump( (encoder, decoder) , f)
	
	print(f'Epoch {epoch}, AALoss: {total_loss_x:.4f}, Edge Loss: {total_loss_edge:.4f}, vq Loss: {total_vq:.4f} , foldx Loss: {total_foldx:.4f} , fapeloss: {total_fapeloss:.4f} , angleloss: {total_angleloss:4f}' )
	writer.add_scalar('Loss/AA', total_loss_x, epoch)
	writer.add_scalar('Loss/Edge', total_loss_edge, epoch)
	writer.add_scalar('Loss/VQ', total_vq, epoch)
	writer.add_scalar('Loss/Foldx', total_foldx, epoch)
	writer.add_scalar('Loss/Fape', total_fapeloss, epoch)
	writer.add_scalar('Loss/Angle', total_angleloss, epoch)

	#log learning rate
	writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)


	total_loss_x = 0
	total_loss_edge = 0
	total_vq = 0
	total_foldx = 0
	total_fapeloss = 0
	total_angleloss = 0

torch.save(encoder.state_dict(), encoder_save)
torch.save(decoder.state_dict(), decoder_save)
with open( modelname+'.pkl', 'wb') as f:
	pickle.dump( (encoder, decoder) , f)
