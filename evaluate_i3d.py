import argparse
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utl
from torch.autograd import Variable
from torchvision import transforms
from pytorch_i3d import InceptionI3d
from VidorDataset_full import VidorDataset
from tqdm.notebook import tqdm 
from util import *
import videotransforms
import time
from apmeter import APMeter
import matplotlib.pyplot as plt
apm = APMeter()
DATASET_LOC = '/home/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'
LOAD_MODEL_LOC = 'models/050.pt'

def load_data(dataset_path, batch_size=1):
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset_train = VidorDataset(dataset_path, 'training', test_transforms)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False,
                                                pin_memory=True)
    dataset_val = VidorDataset(dataset_path, 'validation', test_transforms)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False,
                                                pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    datasets = {'train': dataset_train, 'val': dataset_val}
    return datasets, dataloaders

def compute_acc(outputs,targets):
    pred = np.argmax(outputs[:,1:],1)
    targ = np.argmax(targets,1)
    plt.plot(targ)
    plt.plot(pred)
    print(outputs.shape)
    print(targets.shape)
    print(f'mean_pred:{np.mean(pred)}')
    print(f'mean_targ:{np.mean(targ)}')
    duration = len(pred)
    count = 0
    for i in range(0,duration):
        if(pred[i]==targ[i]):
            count +=1
    print(count/duration)
    return count/duration


datasets, dataloaders = load_data(DATASET_LOC)
num_classes = 42
i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
i3d.replace_logits(num_classes)
i3d.load_state_dict(torch.load(LOAD_MODEL_LOC))
i3d.cuda()

i3d.train(False)
for data in dataloaders['val']:
    # get the inputs
    inputs, labels, feature_path, nf = data
    if nf > 1000:
        preds = []
        for start in range(0,nf,1000):
            end = start + 1000
            with torch.no_grad():
                inputs_variable = Variable(inputs[:,:,start:end].cuda())
                print(inputs_variable.size())
                per_frame_logits = i3d(inputs_variable)
                print(per_frame_logits.size())
                per_frame_logits = F.interpolate(per_frame_logits, inputs_variable.size(2), mode='linear',align_corners=True)
                per_frame_logits = F.softmax(per_frame_logits.squeeze(0).permute(1,0),1)
                preds.append(per_frame_logits.data.cpu().numpy())
        outputs = np.concatenate(preds, axis=0)
    else:
        with torch.no_grad():
            inputs_variable = Variable(inputs.cuda())
            per_frame_logits = i3d(inputs_variable)
            per_frame_logits = F.softmax(per_frame_logits.squeeze(0).permute(1,0),1)
            outputs = per_frame_logits.data.cpu().numpy()
    targets = labels.squeeze(0).permute(1,0).data.cpu().numpy()
    compute_acc(outputs,targets)
    break