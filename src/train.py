import os, sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau
from model import *
from dataset import *
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(note, device, model, dataset_train, dataset_test, batch_size, lr_init, weight_decay, epochs, writer):
    # dataloader
    dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size, shuffle=False, drop_last=True)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    # transfer model to GPU
    model.cuda(device)
    # Loss
    #criterion = nn.MSELoss()
    criterion = chamfer_distance

    ##### BEGINING OF THE EPOCH #####
    t1 = tqdm(range(epochs), desc='Epoch', disable=False)
    for ep in t1:
        train_loss = 0.0
        model.train()
        # training loop
        t2 = tqdm(dataloader_train, desc='Training Step', leave=False)
        for batch in t2:
            # use dataloader to read data
            X, Y =  batch['img'].cuda(device), batch['pcd'].cuda(device)
            # forward prediction
            pred_Y = model(X)
            # calculate loss
            loss, _ = criterion(pred_Y.squeeze(), Y.squeeze())

            # backpropagation
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

            train_loss += loss.item()
            t2.set_postfix(train_step_loss=loss.item())
            
        # validation loop
        # with torch.no_grad():
        valid_loss = 0.0
        model.eval()
        t2 = tqdm(dataloader_test, desc='Validation Step', leave=False)
        for batch in t2:
            # use dataloader to read data
            X, Y =  batch['img'].cuda(device), batch['pcd'].cuda(device)
            # forward prediction
            pred_Y = model(X)
            # calculate loss
            loss, _ = criterion(pred_Y.squeeze(), Y.squeeze())

            valid_loss += loss.item()
            t2.set_postfix(val_step_loss=loss.item())

         ##### ENDING OF THE EPOCH #####
        t1.set_postfix(train_epoch_loss=train_loss/len(dataloader_train), valid_epoch_loss=valid_loss/len(dataloader_test))
        writer.add_scalars('Loss/combine', {'train': train_loss / len(dataloader_train),
                                           'valid': valid_loss / len(dataloader_test)}, ep + 1)
        
        # save model every N epoch
        if (ep+1) % 20 == 0:
            torch.save(model.state_dict(), f'checkpoint/{note}_{ep+1}.pth')    

if __name__ == "__main__":
    
    # os.chdir(sys.path[0])
    
    # input arguments parser
    parser = argparse.ArgumentParser(description="Training Parameters ...")
    
    # additional parameters
    parser.add_argument('--note', default='Dev', type=str, help='additional note to pass into the program')
    parser.add_argument('--device', default=[0], type=int, nargs='+', required=True, help='index of CUDA device')
    parser.add_argument('--epoch', default=100, type=int, required=True, help='number of epoches for training')
    parser.add_argument('--batch_size', default=10, type=int, required=True, help='batch size for each epoch')
    parser.add_argument('--lr_init', default=1e-3, type=float, required=True, help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, required=True, help='weight decay')
    parser.add_argument('--train_dataset', default='', type=str, required=True, help='path to the training dataset')
    parser.add_argument('--valid_dataset', default='', type=str, required=True, help='path to the validation dataset')
    
    args = parser.parse_args()
    
    note = args.note
    device_n = args.device
    random_seed = int(time.time())
    nEpoch = args.epoch
    nBatch = args.batch_size
    lr_init = args.lr_init
    weight_decay = args.weight_decay
    train_dataset = args.train_dataset
    valid_dataset = args.valid_dataset
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    print(f"\n#################### DeepSoRo_Sim2Real Training Script ####################")
    print(f"#### {note} ####")
    # check CUDA and GPU
    print(f"\n########## System Check ##########")
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda', device_n[0])
    print(f'Using CUDA Device: {device_n}')
    print(f"##################################\n")

    writer = SummaryWriter("../runs/"+note)

    model = DeepSoRoNet_VGG(device)
    print('Creating Training Dataset ...')
    dataset_train = DeepSoRoNet_Dataset(train_dataset)
    print('Creating Validation Dataset ...')
    dataset_test = DeepSoRoNet_Dataset(valid_dataset)
    train(note, device, model, dataset_train, dataset_test, batch_size=nBatch, lr_init=lr_init, weight_decay=weight_decay, epochs=nEpoch, writer=writer)

    writer.close()