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
from VidorDataset import VidorDataset
from tqdm import tqdm
from util import *
import videotransforms
DATASET_LOC = '/home/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'

def load_data(dataset_path, batch_size=5, num_workers=10):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset_train = VidorDataset(dataset_path, 'training', train_transforms)
    cls_weights = dataset_train.get_weights()
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                pin_memory=True)
    dataset_val = VidorDataset(dataset_path, 'validation', test_transforms)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=num_workers,
                                                pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    datasets = {'train': dataset_train, 'val': dataset_val}
    return datasets, dataloaders, np.asarray(1-cls_weights, dtype=np.float32)

def print_log(line):
    logging.info(line)
    print(line)

def compute_accuracy(outputs, labels):
    target = torch.max(labels, dim=1)[1]
    pred = torch.max(outputs, dim=1)[1]
    acc = ((target  == pred).sum().data.cpu().numpy() - (target==0).sum().data.cpu().numpy()) / (target.size()[0]*target.size()[1])
    return acc

def run(dataloaders, cls_weights, save_model='output/', num_epochs = 20, num_per_update = 2, num_classes=42):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    # i3d.load_state_dict(torch.load('output-35/039.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    optimizer = optim.SGD(i3d.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-7)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    iteration = 0
    max_iterations = num_epochs * len(dataloaders['train'])
    best_loss = 100
    for epoch in range(num_epochs):
        # Training phase
        print_statement('MODEL TRAINING')
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        for data in tqdm(dataloaders['train']):
            iteration+=1
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)
            labels = Variable(labels.cuda())
            cls_weights = Variable(cls_weights.cuda())
            # set the model to train
            i3d.train(True)
            optimizer.zero_grad()
            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits.permute(0,2,1), labels.permute(0,2,1))
            tot_loc_loss += loc_loss.data.item()
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.data.item()
            # compute total loss
            loss = (0.5*loc_loss + 0.5*cls_loss)
            tot_loss += loss.data.item()
            loss.backward()
            if iteration % num_per_update == 0:
                optimizer.step()
                lr_sched.step()
                if iteration % 10 == 0:
                    acc = compute_accuracy(per_frame_logits,labels)
                    print_log('Epoch:{:d}, Train step:{:d}/{:d},Loc Loss: {:.4f},Cls Loss: {:.4f},Tot Loss: {:.4f}, ACC:{:.4f}'
                          .format(epoch,iteration,max_iterations,tot_loc_loss/num_per_update,tot_cls_loss/num_per_update,tot_loss/num_per_update,acc))
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
        
        torch.save(i3d.module.state_dict(), save_model+str(epoch).zfill(3)+'.pt')    
        # Validation phase
        print_statement('MODEL VALIDATING')
        # set the model to evaluate
        i3d.train(False)
        optimizer.zero_grad()
        tot_loss = tot_loc_loss = tot_cls_loss = 0.
        num_iter = 0 
        overall_acc = 0
        for data in tqdm(dataloaders['val']):
            num_iter+=1
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)
            labels = Variable(labels.cuda())

            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.data.item()
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.data.item()
            # compute total loss
            loss = (0.5*loc_loss + 0.5*cls_loss)
            tot_loss += loss.data.item()

            overall_acc += compute_accuracy(per_frame_logits,labels)
        print_log('Validation: Loc Loss: {:.4f},Cls Loss: {:.4f},Tot Loss: {:.4f}'
                .format(tot_loc_loss/num_iter,tot_cls_loss/num_iter,tot_loss/num_iter))
        print_log('Overall Accuracy:{:.4f}'.format(overall_acc/num_iter))

            
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="output/train.log", filemode="w+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    datasets, dataloaders, cls_weights = load_data(DATASET_LOC)
    run(dataloaders,torch.from_numpy(cls_weights))
