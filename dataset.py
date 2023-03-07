import os, sys
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision

class DeepSoRoNet_Dataset(Dataset):

    def __init__(self, path):
        super(DeepSoRoNet_Dataset, self).__init__()

        self.data = np.load(path)
        self.img = self.data['imgs']
        self.pcd = self.data['pcds']
        self.resize = torchvision.transforms.Resize(224)
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return {
            'img': torch.FloatTensor(self.img[idx]),
            'pcd': torch.FloatTensor(self.pcd[idx]),
        }

