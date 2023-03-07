import os, sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from model import *
from dataset import *
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


criterion = chamfer_distance

def eval(device, model, dataset, frames):
    # DeepSoRoNet
    dataloader = DataLoader(dataset, batch_size=frames, shuffle=True, drop_last=True)
    
    model.cuda(device)
    model.eval()
    batch = next(iter(dataloader))
    X, Y = batch['img'].cuda(device), batch['pcd'].cuda(device)
    pred_Y = model(X)

    loss, _ = criterion(pred_Y.squeeze(), Y.squeeze())
    # backpropagation
    loss.backward()

    gradients = model.get_activations_gradient()
    # print(f'gradients: {pred_Y.size()}')

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activation(X).detach()

    # weight the channels by corresponding gradients
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]

    # print(f'activations: {activations.size()}')
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # print(f'heatmap: {heatmap.size()}')

    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    pred_Y = pred_Y.detach().squeeze().cpu().numpy()
    gt_Y = Y.squeeze().cpu().numpy()
    IMG = X.squeeze().cpu().numpy()
    
    return loss, IMG, pred_Y, gt_Y, heatmap
    
if __name__ == '__main__':
    
    # os.chdir(sys.path[0])
    
    print(f"\n########## System Check ##########")
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    parser = argparse.ArgumentParser(description="Inference Parameters ...")
    
    parser.add_argument('--dataset', default='Dev', type=str, help='path to inference dataset')
    parser.add_argument('--checkpoint', default='Dev', type=str, help='path to model checkpoint')
    parser.add_argument('--filename', default='visual_', type=str, required=True, help='number of epoches for training')
    
    args = parser.parse_args()
    ############ USER INPUT ############
    # number of frames for inference
    frames = 40

    # select your gpu device, default ('cuda', 0)
    device = torch.device('cuda', 0)

    # testing data file path
    dataset = args.dataset

    # string that will become part of the output filename
    # you can change dense1/dense2 resnet/vgg according to you selection
    output_name = args.filename

    ## select one of the following model , and comment the other one##
    model = DeepSoRoNet_VGG(device)

    # path to the model file
    model.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0'))
    
    ############ USER INPUT END ############

    dataset = DeepSoRoNet_Dataset(dataset)

    loss, IMG, pred_Y, gt_Y, heatmap= eval(device, model, dataset, frames)
    print(f'Loss: {loss}')
    heatmap = heatmap.numpy()
    print(f'IMG: {np.shape(IMG)}')
    print(f'Pred Y: {np.shape(pred_Y)}')
    print(f'GT Y: {np.shape(gt_Y)}')
    print(f'Heatmap: {np.shape(heatmap)}')

    fig = plt.figure(figsize=(8, 6))
    ax_img = fig.add_subplot(231)
    ax_act = fig.add_subplot(232)
    ax_sup = fig.add_subplot(233)

    ax_gt = fig.add_subplot(234, projection='3d')
    ax_pred = fig.add_subplot(235, projection='3d')
    ax_mix = fig.add_subplot(236, projection='3d')
    
    def animate(i):
        print(f'Processing Frame: {i}')
        
        ax_img.clear()
        ax_img.imshow(IMG[i], cmap='gray', interpolation='nearest')
        ax_img.set_title(f'Embedded Observation: {i}')

        ax_act.clear()
        ax_act.imshow(heatmap[i], cmap='jet', interpolation='nearest')
        ax_act.set_title(f'Heatmap: {i}')

        ax_sup.clear()
        tmp = cv2.resize(heatmap[i], (IMG[i].shape[1], IMG[i].shape[0]))
        tmp = np.uint8(255 * tmp)
        tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
        img = cv2.cvtColor(IMG[i], cv2.COLOR_GRAY2RGB)
        # print(tmp.dtype, img.dtype)
        # print(tmp, img)
        superimposed_img = cv2.addWeighted(tmp.astype(np.uint8), 0.5, img.astype(np.uint8), 0.1, 0)
        ax_sup.imshow(superimposed_img)


        ax_gt.clear()
        ax_gt.scatter(gt_Y[i, :, 0], gt_Y[i, :, 1], gt_Y[i, :, 2], s=0.1)
        ax_gt.set_title(f'Ground Truth: {i}')
        value = 60
        ax_gt.set_xlim(-value, value); ax_gt.set_ylim(-value, value); ax_gt.set_zlim(-value, value)
        
        ax_pred.clear()
        ax_pred.scatter(pred_Y[0, :, 0], pred_Y[0, :, 1], pred_Y[0, :, 2], s=0.00001)
        ax_pred.scatter(pred_Y[i, :, 0], pred_Y[i, :, 1], pred_Y[i, :, 2], s=0.1)
        ax_pred.set_title(f'Estimation on Testing Data: {i}')
        ax_pred.set_xlim(-value, value); ax_pred.set_ylim(-value, value); ax_pred.set_zlim(-value, value)

        ax_mix.clear()
        ax_mix.scatter(pred_Y[i, :, 0], pred_Y[i, :, 1], pred_Y[i, :, 2], s=0.1)
        ax_mix.scatter(gt_Y[i, :, 0], gt_Y[i, :, 1], gt_Y[i, :, 2], s=0.02)
        ax_mix.set_xlim(-value, value); ax_mix.set_ylim(-value, value); ax_mix.set_zlim(-value, value)
        
    ani = animation.FuncAnimation(fig, animate, frames=frames)
    ani.save(f'results/{output_name}.gif', fps=int(frames/40))

    