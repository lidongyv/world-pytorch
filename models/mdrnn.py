"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
import numpy as np

def lognormal(y, mu, sigma):
    logSqrtTwoPI = torch.log(torch.sqrt(2.0 *torch.ones(1)* np.pi)).cuda(y.device)
    #print(y.shape,mu.shape)
    return -0.5 * torch.pow((y - mu) / torch.exp(sigma),2) - sigma - logSqrtTwoPI


def get_lossfunc(pi, mu, sigma, y):
    v = pi + lognormal(y, mu, sigma)
    #print('v1', torch.max(v),torch.min(v))
    v = torch.log(torch.max(torch.sum(torch.exp(v), dim=-1, keepdim=True),1e-4*torch.ones(1).cuda(pi.device)))
    #print('v2', torch.min(torch.sum(torch.exp(v), dim=-1, keepdim=True)))
    #print('v3', torch.mean(v))
    return -torch.mean(v)


def gmm_loss(mus, sigmas, pi,output):

    logpi=pi-torch.log(torch.sum(torch.exp(pi),dim=-1,keepdim=True))
    print('logpi',torch.mean(logpi))
    logpi=logpi.view(-1,logpi.shape[-1])
    mus = mus.view(-1, mus.shape[-1])
    sigmas = sigmas.view(-1,sigmas.shape[-1])
    # reshape target data so that it is compatible with prediction shape
    flat_target_data = output.contiguous().view(-1,1)

    loss = get_lossfunc(logpi, mus, sigmas, flat_target_data)

    return loss

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, 3* latents * gaussians)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, latents,train=True):
        if train:
            self.rnn.flatten_parameters()
        latents=latents.transpose(1,0)
        seq_len=latents.shape[0]
        bs=latents.shape[1]
        outs, _ = self.rnn(latents)
        # print(outs.shape)
        # exit()
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.latents, self.gaussians)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.latents, self.gaussians)
        # sigmas = torch.exp(sigmas)
        pi = gmm_outs[:, :, 2 * stride: 3 * stride]
        #print('pi',pi.shape,'mus',mus.shape,'sigmas',sigmas.shape)
        pi = pi.view(seq_len, bs, self.latents, self.gaussians)
        # logpi = f.log_softmax(pi, dim=-1)


        return mus, sigmas, pi
class MDRNNCell(_MDRNNBase):

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden):
        print(action.shape,latent.shape)
        in_al = torch.cat([latent,action], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        pi = f.softmax(pi, dim=-1)

        return mus, sigmas, pi, next_hidden
