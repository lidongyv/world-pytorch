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
from models.vae import VAE
import visdom
import cv2
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders_pt import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=25*8, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
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

trained=0
#model = VAE(3, LSIZE).to(device)
model=VAE(3, LSIZE)
model=torch.nn.DataParallel(model,device_ids=range(8))
model.cuda()
model.eval()
# vis = visdom.Visdom(env='vae_pt')
#
# ground_window = vis.image(
#     np.random.rand(RED_SIZE*10, RED_SIZE*10),
#     opts=dict(title='ground!', caption='ground.'),
# )
# image_window = vis.image(
#     np.random.rand(RED_SIZE*10, RED_SIZE*10),
#     opts=dict(title='image!', caption='image.'),
# )
# vae_window = vis.image(
#     np.random.rand(RED_SIZE*10, RED_SIZE*10),
#     opts=dict(title='vae!', caption='vae.'),
# )
# error_window = vis.image(
#     np.random.rand(RED_SIZE*10, RED_SIZE*10),
#     opts=dict(title='error!', caption='error.'),
# )

# check vae dir exists, if not, create it
rnn_dir = join(args.logdir, 'mdrnn')
if not exists(rnn_dir):
    mkdir(rnn_dir)
    mkdir(join(rnn_dir, 'samples'))
vae_dir = join(args.logdir, 'vae')
reload_file = join(vae_dir, 'best.pkl')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
print('loading rnn data')
data=np.load('./log/mdrnn/sample/rnn_data.npz')
mu_record=torch.from_numpy(data['mu'][:20]).cuda()
sigma_record=torch.from_numpy(data['sigma'][:20]).cuda()
pi_record=torch.from_numpy(data['pi'][:20]).cuda()
ground_record=torch.from_numpy(data['ground'][:20]).cuda()
print('loading vae data')
vae_images=np.load('./log/mdrnn/sample/vae_data_t.npz')['raw']
#print(vae_images.shape)
#(20, 1000, 64, 64, 3)

# print(mu.shape)
#torch.Size([2000, 999, 32, 5])
# exit()
print('predict')
with torch.no_grad():
    for i in range(mu_record.shape[0]):
        vae_step=vae_images[i][1:]
        vae_in=vae_images[i][:-1]
        sigma = sigma_record[i]
        epsilon = torch.randn_like(sigma)
        z=torch.sum(pi_record[i]*(mu_record[i]+sigma*epsilon),dim=-1)
        predict = np.reshape(model.module.decoder(z).data.cpu().numpy().astype('float32'),[sigma.shape[0], 3, RED_SIZE, RED_SIZE])
        ground = np.reshape(model.module.decoder(ground_record[i]).data.cpu().numpy().astype('float32'),[sigma.shape[0], 3, RED_SIZE, RED_SIZE])
        print('visualization')
        # ground_gif=[]
        # predict_gif=[]
        gif=[]
        for j in range(predict.shape[0]):
            # if j%10!=0:
            #     continue
            # time.sleep(0.03)
            #print(255*np.mean(np.abs(ground[j, ...]-predict[j,...])))
            ground_image = ground[j, ...]
            image = predict[j,...]
            vae_image=vae_step[j,...]
            vae_in_image = vae_in[j, ...]
            ground_image = np.reshape(ground_image, [3, RED_SIZE, RED_SIZE]).transpose(1,2,0)*255
            image = np.reshape(image, [3, RED_SIZE, RED_SIZE]).transpose(1,2,0)*255
            #vae_image = np.reshape(vae_image, [3, RED_SIZE, RED_SIZE]).transpose(1, 2, 0) * 255

            error_image=np.abs(ground_image-image)
            image=np.array(cv2.resize(image, (RED_SIZE*10, RED_SIZE*10), interpolation=cv2.INTER_CUBIC)).transpose(2,0,1)
            ground_image=np.array(cv2.resize(ground_image, (RED_SIZE*10, RED_SIZE*10), interpolation=cv2.INTER_CUBIC)).transpose(2,0,1)
            error_image=np.array(cv2.resize(error_image, (RED_SIZE*10, RED_SIZE*10), interpolation=cv2.INTER_CUBIC)).transpose(2,0,1)
            # ground_gif.append(ground_image.transpose(1,2,0))
            # predict_gif.append(image.transpose(1,2,0))
            vae_image=np.array(cv2.resize(vae_image, (RED_SIZE*10, RED_SIZE*10), interpolation=cv2.INTER_CUBIC)).transpose(2,0,1)
            vae_in_image=np.array(cv2.resize(vae_in_image, (RED_SIZE*10, RED_SIZE*10), interpolation=cv2.INTER_CUBIC)).transpose(2,0,1)

            gif.append(np.concatenate([vae_in_image.transpose(1,2,0),ground_image.transpose(1,2,0)[:,1:10,:]*0,vae_image.transpose(1,2,0),ground_image.transpose(1,2,0)[:,1:10,:]*0, \
                    ground_image.transpose(1,2,0),ground_image.transpose(1,2,0)[:,1:10,:]*0,image.transpose(1,2,0)],axis=1))
            # vis.image(
            #     ground_image,
            #     opts=dict(title='ground!', caption='ground.'),
            #     win=ground_window,
            # )
            # vis.image(
            #     vae_image,
            #     opts=dict(title='vae!', caption='vae.'),
            #     win=vae_window,
            # )
            # vis.image(
            #     image,
            #     opts=dict(title='image!', caption='image.'),
            #     win=image_window,
            # )
            # vis.image(
            #     np.sum(error_image,axis=0)*255,
            #     opts=dict(title='error!', caption='error.'),
            #     win=error_window,
            # )
        print('saving gif')
        if i%1==0:
            imageio.mimsave('./log/mdrnn/sample/' + str(i) + '.gif', gif)
        # imageio.mimsave('./log/mdrnn/sample/'+str(i)+'ground.gif', ground_gif)
        # imageio.mimsave('./log/mdrnn/sample/' + str(i) + 'predict.gif', predict_gif)