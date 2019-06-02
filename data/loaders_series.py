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
N=2000
start=6000
M=1000
O=1000
class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root,transform):
        self.transform=transform
        self._root=root
        self._files = self._create_file(root)
        self._files.sort()

        self._files = self._files[start:start+N]
        self._data = self._create_dataset(self._files, N, M)

        self._length=self.__len__()


    def __len__(self):
        return len(self._data)
    def _create_file(self,root, N=10000, M=1000):  # N is 10000 episodes, M is number of timesteps
        filelist = []
        idx = 0
        first_root=os.listdir(root)
        for i in range(len(first_root)):
            second_root=os.listdir(os.path.join(root,first_root[i]))
            for j in range(len(second_root)):
                filelist.append(os.path.join(root, first_root[i],second_root[j]))
        return filelist
    def _create_dataset(self,filelist, N=10000, M=1000):  # N is 10000 episodes, M is number of timesteps
        data=[]
        raw_data_list=[]

        for i in range(N):
            print(i)
            sequence=[]
            filename = filelist[i]
            #print(filename)
            #t_data=np.load(filename)
            raw_data = np.load(filename)['observations']
            action_data=np.load(filename)['actions']
            l = np.min([M,len(raw_data)])
            if l<1000:
                print(l)
                continue
            image_data=[]
            for j in range(l):
                image_data.append(self.transform(raw_data[j]))
            #print(image_data[0].shape,action_data[0]) torch.Size([3, 64, 64]) [-0.4757637   0.45614057  0.68328134]
            data.append([torch.stack(image_data[:l]),action_data[:l]])
            raw_data_list.append(raw_data[:l])

            if i%100==0:
                print("loading file", i + 1,"having length",len(data)*M)
        np.savez_compressed('/home/ld/world-pytorch/log/mdrnn/sample/vae_data.npz', raw=raw_data_list)
        print('done_save')
        exit()
        return data
    def __getitem__(self, index):
        data = self._data[index]
        return data[0],data[1]
    def _get_data(self, data):
        return data

class RolloutObservationDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of images

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data):
        return self._transform(data)


