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
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=1000*8, metavar='N',
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


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_train = RolloutObservationDataset('./datasets/carracing',
                                          transform_train, train=True)
dataset_test = RolloutObservationDataset('./datasets/carracing',
                                         transform_test, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)

trained=0
#model = VAE(3, LSIZE).to(device)
model=VAE(3, LSIZE)
model=torch.nn.DataParallel(model,device_ids=range(8))
model.cuda()


def train(epoch):
    """ One training epoch """
    model.test()
    train_loss = []
    for batch_idx, data,actions in enumerate(train_loader):
        with torch.no_grad():
            data = data.cuda()
            recon_batch, mu, logvar = model(data)
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                len(data) * batch_idx / len(train_loader)/10,
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, np.mean(train_loss) ))

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

for epoch in range(trained+1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    # scheduler.step(test_loss)
    # earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.pkl')
    filename = join(vae_dir, 'checkpoint_'+str(epoch)+'.pkl')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        # 'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if not args.nosamples:
        print('saving image')
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).cuda()
            sample = model.module.decoder(sample).cpu()
            save_image(np.reshape(sample,[RED_SIZE, 3, RED_SIZE, RED_SIZE]),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    # if earlystopping.stop:
    #     print("End of Training because of early stopping at epoch {}".format(epoch))
    #     break
