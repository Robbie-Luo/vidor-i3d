import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import pickle
import random
import cv2
import torch
from torchvision import transforms
import torch.utils.data as data_utl
import videotransforms
from dataset import VidOR
from frames import load_vidor_dataset

class VidorDataset(data_utl.Dataset):

    def __init__(self, dataset_path, split, transforms, mode = 'rgb', low_memory=True, num_classes = 42):
        self.anno_rpath = os.path.join(dataset_path,'annotation')
        self.video_rpath = os.path.join(dataset_path,'video')
        self.frame_rpath = os.path.join(dataset_path,'frame')
        self.split = split
        self.max_length = 60
        self.mode = mode
        self.low_memory = low_memory
        self.num_classes = num_classes
        self.data = self.make_vidor_data()
        self.transforms = transforms

        
    def make_vidor_data(self):
        pkl_path = f'dataset/vidor_{self.split}_data.pkl'
        if os.path.exists(pkl_path):
            vidor_dataset = load_vidor_dataset()
            vidor_data = []
            actions = vidor_dataset._get_action_predicates()
            vids = vidor_dataset.get_index(self.split)
            for ind in tqdm(vids):
                video_path = vidor_dataset.get_video_path(ind,self.split)
                for each_ins in vidor_dataset.get_action_insts(ind):
                    start_f, end_f = each_ins['duration']
                    action = actions.index(each_ins['category'])
                    if end_f - start_f < self.max_length:
                        continue
                    label = np.full((1, end_f - start_f), action)
                    vidor_data.append((video_path, label, start_f, end_f))
            with open(pkl_path,'wb') as file:
                pickle.dump(vidor_data,file)
        else:
            with open(pkl_path,'rb') as file:
                vidor_data = pickle.load(file)
        return vidor_data

    def get_weights(self):
        pkl_path = 'dataset/cls_weights.pkl'
        if not os.path.exists(pkl_path):
            vidor_dataset = load_vidor_dataset()
            actions = vidor_dataset._get_action_predicates()
            vids = vidor_dataset.get_index(self.split)
            action_counts = np.zeros(self.num_classes)
            for ind in tqdm(vids):
                video_path = vidor_dataset.get_video_path(ind,self.split)
                for each_ins in vidor_dataset.get_action_insts(ind):
                    start_f, end_f = each_ins['duration']
                    action = actions.index(each_ins['category'])
                    action_counts[action] += 1
            cls_weights = action_counts/np.sum(action_counts)
            with open(pkl_path,'wb') as file:
                pickle.dump(cls_weights,file)
        else:
            with open(pkl_path,'rb') as file:
                cls_weights = pickle.load(file)
        return cls_weights


    
    def load_rgb_frames(self, video_path, start, end):
        frame_dir = video_path.replace('video','frame').replace('.mp4','')
        frames = []
        for i in range(start, end):
            img_path = os.path.join(frame_dir, str(i+1).zfill(4)+'.jpg')
            assert os.path.exists(img_path)
            img = cv2.imread(img_path)[:, :, [2, 1, 0]]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)
            
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video_path, label, start_f, end_f = self.data[index]
        nf = end_f - start_f
        start_f = random.randint(0,nf - self.max_length)
        imgs = self.load_rgb_frames(video_path, start_f, start_f+  self.max_length)
        label = label[:, start_f:start_f + self.max_length]
        imgs = self.transforms(imgs)
        label = np.asarray((np.eye(self.num_classes)[np.squeeze(label,0)].T),dtype=np.float32)
        frames_tensor  = torch.from_numpy(imgs.transpose([3, 0, 1, 2]))
        return frames_tensor, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    dataset_path ='/home/wluo/vidor-dataset'
    dataset = VidorDataset(dataset_path, 'training')
    dataset = VidorDataset(dataset_path, 'validation')
    


    


