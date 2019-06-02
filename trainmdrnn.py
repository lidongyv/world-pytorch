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
mdrnn=torch.nn.DataParallel(mdrnn,device_ids=[1,2,3,4,5,6,7])
mdrnn.cuda(1)
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


# Data Loading
train_loader = DataLoader(
    _RolloutDataset('datasets/carracing2_rnn',train=True),
    batch_size=BSIZE, num_workers=8, shuffle=True)
test_loader = DataLoader(
    _RolloutDataset('datasets/carracing2_rnn',train=False),
    batch_size=BSIZE, num_workers=8)

def get_loss(input, output,train):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    mu,sigma,pi = mdrnn(input,train)
    gmm = gmm_loss(mu,sigma,pi,output)

    return gmm
def generate_latent(data):
    mu=data[...,:LSIZE]
    sigmas=data[...,LSIZE:2*LSIZE]
    actions=data[...,2*LSIZE:]
    sigma = torch.exp(sigmas / 2.0)
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon
    return torch.cat([z,actions],dim=-1)
def data_pass(epoch, train):
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    cum_loss = 0
    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        #data=generate_latent(data.cuda(1)).transpose(1,0)
        #data=data.cuda(1).transpose(1,0)
        data=data.cuda(1)
        input=data[:,:-1,:]
        output = data[:,1:,:32]
        #print('input.shape',input.shape)
        if train:
            for j in range(30):
                if 33*(j+1)>input.shape[1]:
                    losses = get_loss(input[:,-33:,:],output[:,-33:,:], train)
                else:
                    losses = get_loss(input[:,33*j:33*(j+1),:],output[:,33*j:33*(j+1),:],train)
                print('epoch',epoch,'batch',i,'split',j,'loss:',losses.item(),'\n')
                optimizer.zero_grad()
                losses.backward()
                if torch.isinf(losses) or torch.isnan(losses):
                    print('error')
                    continue
                optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(input,output,train)
                if torch.isinf(losses) or torch.isnan(losses):
                    print('error')
                    exit()
                print('losses',losses)

        cum_loss += losses.item()
        print(cum_loss)


        pbar.set_postfix_str("loss={loss:10.6f}".format(
                                 loss=cum_loss / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True)
test = partial(data_pass, train=False)

cur_best = None
for e in range(epochs):

    print('train')
    train(e)
    print('test')
    test_loss = test(e)
    # scheduler.step(test_loss)
    # earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint'+str(e)+'.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        # 'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)

