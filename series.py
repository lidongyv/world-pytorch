""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from models.vae import VAE
import visdom
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders_series import _RolloutDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', default='log_pt',type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()
learning_rate=1e-4

torch.manual_seed(914)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])
trained=0
#model = VAE(3, LSIZE).to(device)
model=VAE(3, LSIZE)
model=torch.nn.DataParallel(model,device_ids=range(8))
model.cuda()

dataset_train = _RolloutDataset('./datasets/carracing2',transform_train)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=64,drop_last=False)




def train(epoch):
    """ One training epoch """
    model.eval()
    record_mu=[]
    record_logvar=[]
    record_actions=[]
    for batch_idx, [observation,actions] in enumerate(train_loader):
        with torch.no_grad():
            print('actions',actions.shape)
            print('observation',observation.shape)
            observation = observation.cuda().view(observation.shape[0]*observation.shape[1],3,64,64)
            _,mu, logvar = model(observation)

            record_logvar.append(logvar.view(observation.shape[0],32).data.cpu().numpy().astype('float32'))
            record_mu.append(mu.view(observation.shape[0],32).data.cpu().numpy().astype('float32'))
            record_actions.append(actions.view(observation.shape[0],3).numpy().astype('float32'))
            print(batch_idx,len(record_logvar))
            # print('actions_record',actions.shape)
            # print('observation_record',observation.shape)

    record_logvar=np.reshape(np.concatenate(record_logvar),[-1,1000,32])
    record_mu=np.reshape(np.concatenate(record_mu),[-1,1000,32])
    record_actions=np.reshape(np.concatenate(record_actions),[-1,1000,3])
    print(record_logvar.shape,record_mu.shape,record_actions.shape)
    np.savez_compressed(join('./datasets/carracing2_rnn', 'rnn_data.npz'),
                        logvar=record_logvar,
                        mu=record_mu,
                        actions=record_actions)

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.pkl')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    trained=state['epoch']
    #trained=0
    # scheduler.load_state_dict(state['scheduler'])
    # earlystopping.load_state_dict(state['earlystopping'])


cur_best = None
train(trained)
