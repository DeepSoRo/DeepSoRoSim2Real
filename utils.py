import os, sys
import torch
import torch.nn as nn
import torchvision.models.resnet as ResNet
import torchvision.models.vgg as VGGNet
from torchinfo import summary

class MLP(nn.Sequential):

    def __init__(self, dims, doLastRelu=False):
            layers = self.getMLPLayers(dims, doLastRelu)
            super(MLP, self).__init__(*layers)  

    def get_and_init_FC_layer(self, din, dout):
        li = nn.Linear(din, dout)
        nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
        li.bias.data.fill_(0.)

        return li

    def getMLPLayers(self, dims, doLastRelu):
        layers = []
        for i in range(1, len(dims)):
            layers.append(self.get_and_init_FC_layer(dims[i-1], dims[i]))
            if i == len(dims)-1 and not doLastRelu:
                continue
            layers.append(nn.ReLU())
            
        return layers

class VGG11(nn.Module):
    def __init__(self, in_channels):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.gradients = None

        # convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
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

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv_layers(x)
        if self.training:
            h = x.register_hook(self.activations_hook)
        # print(f'Conv Layer Output: {x.size()}')
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=7).squeeze()
        # print(f'VGG Output: {x.size()}')

        return x

    # method for gradient extraction
    def get_activation_gradient(self):
        return self.gradients

    # method for activation extraction
    def get_activation(self, x):
        return self.conv_layers(x)



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        ##### ResNet #####
        resnet = ResNet.resnet18()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.avgpool = resnet.avgpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        print(f'0: {x.size()}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f'0.5: {x.size()}')
        x = self.layer1(x)
        print(f'1: {x.size()}')
        x = self.layer2(x)
        print(f'2: {x.size()}')
        x = self.layer3(x)
        print(f'3: {x.size()}')
        x = self.layer4(x)
        print(f'4: {x.size()}')
        x = self.avgpool(x).squeeze(-1).squeeze(-1)

        return x

class DeepSoRo_Decoder(nn.Module):
    
    def __init__(self, mlp_dims, fold_dims, prototype):
        super(DeepSoRo_Decoder, self).__init__()

        self.mlp = MLP(mlp_dims)
        self.fold = MLP(fold_dims)
        self.prototype = prototype
        self.prototype_dim = self.prototype.shape[0]

    def forward(self, global_codeword):
        # expand codeword
        global_codeword = global_codeword.unsqueeze(1)                          # B x 512 -> B x 1 x 512
        global_codeword = global_codeword.expand(-1, self.prototype_dim, -1)    # B x 1 x 512 -> B x V x 512

        # expand prototype to the same dimension of the codeword
        prototype = self.prototype.unsqueeze(0)                                 # V x 3 -> 1 x V x 3
        prototype = prototype.expand(global_codeword.shape[0], -1, -1)          # 1 x V x 3 -> B x V x 3
        prototype = self.mlp(prototype)                                         # B x V x 3 -> B x V x 512

        # concat prototype and codeword
        combined = torch.cat((prototype, global_codeword), dim=2)               # B x V x 512+512
        x = self.fold(combined)

        return x

if __name__ == '__main__':

    os.chdir(sys.path[0])
    device = torch.device('cuda', 0)
    model = CNN()
    model.cuda(device)
    
    summary(model, [10, 1, 256, 256])