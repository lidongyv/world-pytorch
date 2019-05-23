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
N=5000
M=1000
O=1000
class _RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, transform, train=True): # pylint: disable=too-many-arguments
        self._transform = transform
        self._root=root
        self._files = self._create_file(root)
        self._files.sort()
        if train:
            self._files = self._files[:N]
            self._data = self._create_dataset(self._files, N, M)
        else:
            self._files = self._files[-O:]
            self._data = self._create_dataset(self._files, O, M)

        self._length=self.__len__()


    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        # if not self._cum_size:
        #     self.load_next_buffer()
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
        for i in range(N):
            filename = filelist[i]
            #print(filename)
            raw_data = np.load(filename)['observations']

            l = len(raw_data)
            #print(l)
            for j in range(l):
                #print(np.max(raw_data[j]),np.min(raw_data[j]),np.mean(raw_data[j]))
                if np.max(raw_data[j])<255:
                    data.append((raw_data[j]))
                else:
                    data.append((raw_data[j]))
            # raw_data=np.stack(true_data)
            l=len(data)

            if i%100==0:
                print("loading file", i + 1,"having length",l)
            if len(data)>M*N:
                data=np.stack(data)[M*N]
            # if (idx + l) > (M * N):
            #     data = data[0:]
            #     print('premature break')
            #     break
            # if i<12000:
            #     count=0
            #     # for j in range(l):
            #     #     if np.mean(data[j])>10:
            #     #         print(i,j,np.mean(data[j]))
            #     #         count+=1
            #     #         # plt.figure("Image")  # 图像窗口名称
            #     #         # plt.imshow(data[j])
            #     #         # plt.axis('on')  # 关掉坐标轴为 off
            #     #         # plt.title('image')  # 图像题目
            #     #         # plt.show()
            #     #         # break
            #     # print(count)
            #     # count=0
            #     for j in range(l):
            #         if np.mean(data[j])<1:
            #             #print(i,j,np.mean(data[j]))
            #             count+=1
            #             if j%500==0:
            #                 plt.figure("Image")  # 图像窗口名称
            #                 plt.imshow(data[j])
            #                 plt.axis('on')  # 关掉坐标轴为 off
            #                 plt.title('image')  # 图像题目
            #                 plt.show()
            #                 print(os.path.join(self._root, filename))
            #             #break
            #     if count>0:
            #         print(i,count)
            # else:
            #     exit()
            # data[idx:idx + l] = raw_data
            # idx += l

            # if ((i + 1) % 100 == 0):
            #     print("loading file", i + 1)
        return data
    def __getitem__(self, index):
        data = self._data[index]
        return self._get_data(data)

    def _get_data(self, data):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
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

    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)

    NOTE: seq_len < rollout_len in moste use cases

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def __init__(self, root, seq_len, transform, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'terminals')]
        # data is given in the form
        # (obs, action, reward, terminal, next_obs)
        return obs, action, reward, terminal, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len

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
