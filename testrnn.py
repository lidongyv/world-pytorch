""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
from torch import optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau

from data.loaders_rnn import _RolloutDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,default='log',
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constants
BSIZE = 8
SEQ_LEN = 999
epochs = 3000
torch.backends.cudnn.benchmark = True
learning_rate=1e-4
# Loading model

rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn=torch.nn.DataParallel(mdrnn,device_ids=range(8))
mdrnn.cuda()
#mdrnn.to(device)
optimizer = optim.Adam(mdrnn.parameters(),lr=1e-4,betas=(0.9,0.999))
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
# earlystopping = EarlyStopping('min', patience=30)


if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    # scheduler.load_state_dict(state['scheduler'])
    # earlystopping.load_state_dict(state['earlystopping'])


test_loader = DataLoader(
    _RolloutDataset('datasets/carracing2_rnn',train=False),
    batch_size=BSIZE, num_workers=8)

mdrnn.eval()
loader = test_loader

mu_record=[]
sigma_record=[]
pi_record=[]
output_record=[]
for i, data in enumerate(loader):

    data=data.cuda()
    input=data[:,:-1,:]
    output = data[:,1:,:32]
    with torch.no_grad():
        mu,sigma,pi = mdrnn(input,train=False)
        sigma=torch.exp(sigma).transpose(1,0).view(8,999,32,5)
        pi=torch.softmax(pi,dim=-1).transpose(1,0).view(8,999,32,5)
        mu=mu.transpose(1,0).view(8,999,32,5)
        output=output.view(8,999,32)
        mu_record.append(mu.data.cpu().numpy())
        sigma_record.append(sigma.data.cpu().numpy())
        pi_record.append(pi.data.cpu().numpy())
        output_record.append(output.data.cpu().numpy())
mu_record=np.concatenate(mu_record,axis=0)
sigma_record=np.concatenate(sigma_record,axis=0)
pi_record=np.concatenate(pi_record,axis=0)
output_record=np.concatenate(output_record,axis=0)
print(mu_record.shape,output_record.shape)
np.savez_compressed(join('./lo'
                         'g/mdrnn/sample/rnn_data.npz'),
                    sigma=sigma_record,
                    mu=mu_record,
                    pi=pi_record,
                    ground=output_record)
