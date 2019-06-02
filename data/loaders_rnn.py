""" Some data loading utilities """
from bisect import bisect
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt
N=8
M=1000
O=1
class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root,train):
        self._root=root
        self._files = self._create_file(root)
        self._files.sort()
        #print(self._files)
        if train:
            self._files = self._files[:N]
            self._data = self._create_dataset(self._files, N, M)
        else:
            self._files = self._files[-O:]
            self._data = self._create_dataset(self._files, O, M)
        # self._files = self._files[:N]
        # self._data = self._create_dataset(self._files, N, M)
        self._length=self.__len__()
    def __len__(self):
        return len(self._data)
    def _create_file(self,root, N=10000, M=1000):  # N is 10000 episodes, M is number of timesteps
        filelist = []
        idx = 0
        first_root=os.listdir(root)
        for i in range(len(first_root)):
            filelist.append(os.path.join(root, first_root[i]))
        return filelist
    def _create_dataset(self,filelist, N=10000, M=1000):  # N is 10000 episodes, M is number of timesteps
        data=[]
        for i in range(N):
            print(i)
            sequence=[]
            filename = filelist[i]
            #print(filename)
            t_data=np.load(filename)
            # print(filename)
            # exit()
            logvar_data = torch.from_numpy(t_data['logvar'])
            mu_data = torch.from_numpy(t_data['mu'])
            action_data=torch.from_numpy(t_data['actions'])
            #print(image_data[0].shape,action_data[0]) torch.Size([3, 64, 64]) [-0.4757637   0.45614057  0.68328134]
            #latent=mu_data + logvar_data.exp() * torch.randn_like(mu_data)
            sigma = torch.exp(logvar_data / 2.0)
            epsilon = torch.randn_like(sigma)
            latent = mu_data + sigma * epsilon
            data.append(torch.cat((latent,action_data),dim=-1))
            if i%1==0:
                print("loading file", i + 1,"having length",2000*len(data)*M)
        #print(len(data))
        data=torch.cat(data,dim=0)
        #print(data.shape)
        return data
    def __getitem__(self, index):
        data = self._data[index]
        data=self._get_data(data)
        #print(data.shape)
        return data
    def _get_data(self, data):
        return data


