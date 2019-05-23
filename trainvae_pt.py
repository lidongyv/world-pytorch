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

dataset_train = RolloutObservationDataset('./datasets/carracing2',
                                          transform_train, train=True)
dataset_test = RolloutObservationDataset('./datasets/carracing2',
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
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
# earlystopping = EarlyStopping('min', patience=30)

vis = visdom.Visdom(env='vae_pt')

ground_window = vis.image(
    np.random.rand(64, 64),
    opts=dict(title='ground!', caption='ground.'),
)
image_window = vis.image(
    np.random.rand(64, 64),
    opts=dict(title='image!', caption='image.'),
)
loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss',
                                 title='Training Loss',
                                 legend=['Loss']))
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    #BCE = torch.mean(torch.sum(torch.pow(recon_x-x,2),dim=(1,2,3)))
    BCE = F.mse_loss(recon_x,x,reduction='sum')/recon_x.shape[0]

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logsigma - torch.pow(mu,2) - torch.exp(logsigma),dim=1)
    #KLD=torch.max(KLD,torch.ones_like(KLD).cuda()*LSIZE*0.5)
    KLD=torch.mean(KLD)
    # print(KLD.shape,logsigma.shape,mu.shape)
    # exit()
    print(BCE.item(),KLD.item())
    return BCE + KLD


def train(epoch):
    """ One training epoch """
    model.train()
    train_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda()
        # print(torch.max(data))
        # exit()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        #print(loss.item())
        train_loss.append(loss.item())
        optimizer.step()
        ground = data[0,...].data.cpu().numpy().astype('float32')
        ground = np.reshape(ground, [3,64, 64])
        vis.image(
            ground,
            opts=dict(title='ground!', caption='ground.'),
            win=ground_window,
        )
        image = recon_batch[0,...].data.cpu().numpy().astype('float32')
        image = np.reshape(image, [3, 64, 64])
        vis.image(
            image,
            opts=dict(title='image!', caption='image.'),
            win=image_window,
        )
        vis.line(
            X=torch.ones(1).cpu() * batch_idx + torch.ones(1).cpu() * (epoch - trained-1)* args.batch_size,
            Y=loss.item() * torch.ones(1).cpu(),
            win=loss_window,
            update='append')
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                len(data) * batch_idx / len(train_loader)/10,
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, np.mean(train_loss) ))


def test():
    """ One test epoch """
    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            data = data.cuda()
            recon_batch, mu, logvar = model(data)
            test_loss.append(loss_function(recon_batch, data, mu, logvar).item())
            ground = data[0, ...].data.cpu().numpy().astype('float32')
            ground = np.reshape(ground, [3, 64, 64])
            vis.image(
                ground,
                opts=dict(title='ground!', caption='ground.'),
                win=ground_window,
            )
            image = recon_batch[0,...].data.cpu().numpy().astype('float32')
            image = np.reshape(image, [3, 64, 64])
            vis.image(
                image,
                opts=dict(title='image!', caption='image.'),
                win=image_window,
            )
    test_loss =np.mean(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

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
    optimizer.load_state_dict(state['optimizer'])
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
