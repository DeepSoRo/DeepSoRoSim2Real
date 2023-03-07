import torch
import torch.nn as nn
import numpy as np
import os, sys
import torchvision.models.resnet as ResNet
from torchinfo import summary
from utils import *

class DeepSoRoNet_VGG(nn.Module):

    def __init__(self, device):
        super(DeepSoRoNet_VGG, self).__init__()
        
        self.device = device
        self.MLP_dim = [3, 256, 512]
        self.FOLD_dim = [1024, 512, 512, 3]
        self.Prototype = self.prototype()

        # MLP Decoder
        self.mlp_decoder = DeepSoRo_Decoder(self.MLP_dim, self.FOLD_dim, self.Prototype)

        # convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # x - sequence of images - Dim: BxNx1xHxW (Batch x Seq_Length x Single_Channel x Height x Width
        """ VGG """
        x = self.conv_layers(x)
        # print(f'VGG Output Dim: {x.size()}')
        # print(x)

        h = x.register_hook(self.activations_hook)

        # exit()

        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=7).squeeze()

        # print(f'VGG Output Dim: {x.size()}')
        
        """ MLP """
        x = self.mlp_decoder(x)
        # print(f'MLP Output Dim: {x.size()}')
        
        return x

    def activations_hook(self, grad):
        self.gradients = grad
    
    # method for gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for activation extraction
    def get_activation(self, x):
        return self.conv_layers(x)

    def prototype(self):
        prototype = np.load('./prototype_finger.npz')
        prototype = torch.FloatTensor(prototype['prototype'])

        return prototype.cuda(self.device)

if __name__ == '__main__':
    os.chdir(sys.path[0])
    
    device = torch.device('cuda', 0)
    model = DeepSoRoNet_VGG(device)
    model.cuda(device)
    summary(model, (10, 1, 256, 256))
