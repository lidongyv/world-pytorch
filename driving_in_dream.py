""" Training VAE """
import argparse
import time
import imageio
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
import visdom
import cv2
import json
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from models import MDRNNCell, VAE, Controller
parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', default='log',type=str, help='Directory where results are logged')
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

trained=0
#model = VAE(3, LSIZE).to(device)
vae_model=VAE(3, LSIZE)
vae_model=torch.nn.DataParallel(vae_model,device_ids=[7])
vae_model.cuda(7)
vae_model.eval()
mdrnn_model = MDRNNCell(LSIZE, ASIZE, RSIZE, 5)
mdrnn_model=torch.nn.DataParallel(mdrnn_model,device_ids=[7])
mdrnn_model.cuda(7)
mdrnn_model.eval()
controller = torch.nn.DataParallel(Controller(LSIZE, RSIZE, ASIZE)).cuda()

vis = visdom.Visdom(env='dream')
image_window = vis.image(
    np.random.rand(RED_SIZE*10, RED_SIZE*10),
    opts=dict(title='dream!', caption='dream.'),
)

# check vae dir exists, if not, create it
dream_dir = join(args.logdir, 'dream')
vae_dir = join(args.logdir, 'vae')
reload_file = join(vae_dir, 'best.tar')
state = torch.load(reload_file)
print("Reloading model at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']))
vae_model.load_state_dict(state['state_dict'])
mdrnn_dir = join(args.logdir, 'mdrnn')
reload_file = join(mdrnn_dir, 'best.tar')
state = torch.load(reload_file)

print("Reloading model at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']))
# mdrnn_model.load_state_dict(state['state_dict'])
# print(state['state_dict'])
mdrnn_model.load_state_dict(
    {k.strip('_l0'): v for k, v in state['state_dict'].items()})
control=torch.nn.Linear(288,3)
control_weight=json.load(open('./log/ctrl/carracing.cma.16.64.best.json'))[0]
#control weight shape is 867= (256+32)*3+3, not sure how to assign the weight to controller
weights=np.array(control_weight[:-3])
bias=np.array(control_weight[-3:])
control.weight=torch.nn.Parameter(torch.from_numpy(weights).float())
control.bias=torch.nn.Parameter(torch.from_numpy(bias).float())
control=control.cuda(7)
# exit()
print('dreaming')
with torch.no_grad():
    for i in range(20):
        sample = torch.randn(1, LSIZE).cuda(7)
        #print(sample.shape)
        sample_image = vae_model.module.decoder(sample)
        print('visualization')
        gif=[]
        #print(sample_image.shape)
        sample_image_v=np.reshape(sample_image.cpu().numpy().astype('float32'),
                   [3, RED_SIZE, RED_SIZE]).transpose(1,2,0)*255
        sample_image_v = np.array(cv2.resize(sample_image_v, (RED_SIZE * 10, RED_SIZE * 10), interpolation=cv2.INTER_CUBIC))
        gif.append(sample_image_v)
        sample_image_v = sample_image_v.transpose(2, 0, 1)
        vis.image(
            sample_image_v,
            opts=dict(title='image!', caption='image.'),
            win=image_window,
        )
        hidden = [
            torch.zeros(1, RSIZE).cuda(7)
            for _ in range(2)]
        for j in range(1000):
            # sample_image=torch.from_numpy(sample_image/255).cuda(7)
            _, latent_mu, _ = vae_model(sample_image)
            #only mu is used
            #print(latent_mu.shape,hidden[0].shape)
            #action = control(torch.cat([latent_mu, hidden[0]],dim=1).view(-1))
            #print(control.weight.view(288,3).shape,torch.cat([latent_mu, hidden[0]],dim=1).view(288,1).shape,control.bias.shape)
            action=torch.sum(control.weight.view(288,3)*torch.cat([latent_mu, hidden[0]],dim=1).view(288,1),dim=0)+control.bias
            action=action.view(1,3)
            mus, sigmas, pi, hidden = mdrnn_model(action, latent_mu, hidden)
            epsilon = torch.randn_like(sigmas)
            #print(mus.shape,sigmas.shape,pi.shape)
            sample = torch.sum(pi.unsqueeze(-1) * (mus + sigmas * epsilon), dim=1)
            #print(sample.shape)
            sample_image = vae_model.module.decoder(sample)
            sample_image_v = np.reshape(sample_image.cpu().numpy().astype('float32'),
                                      [3, RED_SIZE, RED_SIZE]).transpose(1, 2, 0) * 255
            sample_image_v = np.array(
                cv2.resize(sample_image_v, (RED_SIZE * 10, RED_SIZE * 10), interpolation=cv2.INTER_CUBIC))
            gif.append(sample_image_v)

            sample_image_v=sample_image_v.transpose(2, 0, 1)
            vis.image(
                sample_image_v,
                opts=dict(title='image!', caption='image.'),
                win=image_window,
            )
        print('saving gif')
        if i%1==0:
            imageio.mimsave('./log/mdrnn/sample/' + str(i) + '.gif', gif)
        # imageio.mimsave('./log/mdrnn/sample/'+str(i)+'ground.gif', ground_gif)
        # imageio.mimsave('./log/mdrnn/sample/' + str(i) + 'predict.gif', predict_gif)