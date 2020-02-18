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
from tqdm import tqdm
from util import *
import videotransforms
import time
DATASET_LOC = '/home/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'
LOAD_MODEL_LOC = 'models/050.pt'

def load_data(dataset_path, batch_size=1):
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset_train = VidorDataset(dataset_path, 'training', test_transforms)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=5,
                                                pin_memory=True)
    dataset_val = VidorDataset(dataset_path, 'validation', test_transforms)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=5,
                                                pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    datasets = {'train': dataset_train, 'val': dataset_val}
    return datasets, dataloaders

def print_log(line):
    logging.info(line)
    print(line)

def run(dataloaders,num_classes=42 ):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(LOAD_MODEL_LOC))
    i3d.cuda()
    
    i3d.train(False)
    count = 0
    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode
        # Iterate over data.
        for data in tqdm(dataloaders[phase]):
            # get the inputs
            inputs, labels, feature_path, nf = data
            count += 1
            if os.path.exists(feature_path[0]+'/i3d_040'+'.npy'):
                continue

            os.makedirs(feature_path[0], exist_ok=True)
            b,c,t,h,w = inputs.shape
            print_log(f'shape:{b},{c},{t},{h},{w}')
            print_log('path:'+feature_path[0])
            # b,c,t,h,w = inputs.shape
            print_log(f'count:{count}')
            print_log(f'num_frames:{nf}')
            time_a = time.time()
            if nf > 1000:
                features = []
                for start in range(0, nf, 1000):
                    end = start+1000
                    if end > nf - 1:
                        end = nf - 1
                    with torch.no_grad():
                        ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda())
                        features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(feature_path[0],'i3d_040'), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                with torch.no_grad():
                    inputs = Variable(inputs.cuda())
                    features = i3d.extract_features(inputs)
                    np.save(os.path.join(feature_path[0],'i3d_040'), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            time_b = time.time()
            print_log(f'time consumed:{time_b-time_a}s')
if __name__ == '__main__':
    datasets, dataloaders = load_data(DATASET_LOC)
    logging.basicConfig(level=logging.DEBUG, filename="extract.log", filemode="w+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    run(dataloaders)
