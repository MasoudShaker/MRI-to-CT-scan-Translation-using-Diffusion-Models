# dataset.py DDPM folder
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset


class Train_Data(Dataset):
    def __init__(self):
        path = 'mr_train_resized.hdf5'
        f = h5py.File(path,'r')
        load_data = f['data']
        self.lr = load_data
        path = 'ct_train_resized.hdf5'
        f = h5py.File(path,'r')
        load_data = f['data']
        self.hr = load_data
        c, self.h, self.w = self.lr.shape

        self.len = c

    def __getitem__(self, index):
        x = self.lr[index, :, :]
        y = self.hr[index, :, :]

        x = self.norm(x)
        y = self.norm(y)

        sample = {'lr': x,'hr': y}

        x, y = sample['lr'], sample['hr']

        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))

        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()

        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)

        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)

        return xx, yy

    def __len__(self):
        return self.len

    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x


class Test_Data(Dataset):
    def __init__(self):
        path = 'mr_test_resized.hdf5'
        f = h5py.File(path,'r')
        load_data = f['data']
        self.lr = load_data
        path = 'ct_test_resized.hdf5'
        f = h5py.File(path,'r')
        load_data = f['data']
        self.hr = load_data
        c, self.h, self.w = self.lr.shape

        self.len = c

    def __getitem__(self, index):
        x = self.lr[index, :, :]
        y = self.hr[index, :, :]

        x = self.norm(x)
        y = self.norm(y)

        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))

        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()

        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)

        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)

        return xx, yy

    def __len__(self):
        return self.len

    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x