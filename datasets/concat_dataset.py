import os, sys
import os.path
import cv2
import random
import glob
import torch
import torch.utils.data as data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr


class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.num_datasets = len(datasets)

        # dataset size
        name = ''
        size = 0
        for d in self.datasets:
            size += len(d)
            if name == '':
                name = d._name
            else:
                name = name + '_' + d._name
        self._size = size
        self._name = name
        print(self._name)

        # dataset index
        set_index = np.zeros((size, ), dtype=np.int32)
        data_index = np.zeros((size, ), dtype=np.int32)
        start = 0
        for i in range(self.num_datasets):
            num = len(self.datasets[i])
            set_index[start:start+num] = i
            data_index[start:start+num] = np.arange(num)
            start += num
            print('%d data in dataset %d' % (num, i))

        self._set_index = set_index
        self._data_index = data_index
        

    def __getitem__(self, i):
        ind_set = self._set_index[i]
        ind_data = self._data_index[i]
        return self.datasets[ind_set][ind_data]


    def __len__(self):
        return self._size
