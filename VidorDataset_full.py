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
        self.mode = mode
        self.low_memory = low_memory
        self.num_classes = num_classes
        self.data = self.make_vidor_data()
        self.transforms = None
        
    def make_vidor_data(self):
        pkl_path = f'dataset/vidor_{self.split}_data_full.pkl'
        if not os.path.exists(pkl_path):
            vidor_dataset = load_vidor_dataset()
            vidor_data = []
            actions = vidor_dataset._get_action_predicates()
            vids = vidor_dataset.get_index(self.split)
            for ind in tqdm(vids):
                video_path = vidor_dataset.get_video_path(ind,self.split)
                frame_dir = video_path.replace('video','frame').replace('.mp4','')
                valid_frame_dir = video_path.replace('video','valid_frame').replace('.mp4','')
                num_frames = len(os.listdir(frame_dir))
                num_valid_frames = len(os.listdir(valid_frame_dir))
                label = np.zeros((self.num_classes,num_frames), np.float32)
                for each_ins in vidor_dataset.get_action_insts(ind):
                    start_f, end_f = each_ins['duration']
                    action = actions.index(each_ins['category'])
                    for fr in range(0,num_frames,1):
                        if fr >= start_f and fr < end_f:
                            label[action, fr] = 1 # binary classification
                label = label[:,~np.all(label == 0, axis=0)]
                assert num_valid_frames == label.shape[1]
                vidor_data.append((valid_frame_dir,label,num_valid_frames))
            with open(pkl_path,'wb') as file:
                pickle.dump(vidor_data,file)
        else:
            with open(pkl_path,'rb') as file:
                vidor_data = pickle.load(file)
        return vidor_data

    
    def load_rgb_frames(self, frame_dir):
        frame_paths = sorted(glob.glob(frame_dir+'/*.jpg'))
        frames = []

        for i in range(0, len(frame_paths)):
            # img = cv2.imread(os.path.join(frame_dir, str(i).zfill(4)+'.jpg'))[:, :, [2, 1, 0]]
            assert os.path.exists(frame_paths[i])
            img = cv2.imread(frame_paths[i])[:, :, [2, 1, 0]]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            w,h,c = img.shape
            i = int(np.round((w - 224) / 2.))
            j = int(np.round((h - 224) / 2.))

            img = img[i:i + 224, j:j + 224, :]
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

        frame_dir, label, nf = self.data[index]
        feature_path =  frame_dir.replace('valid_frame','feature')
        
        if os.path.exists(feature_path+'/i3d_040'+'.npy'):
            return 0,0, feature_path , nf

        imgs = self.load_rgb_frames(frame_dir)

        frames_tensor  = torch.from_numpy(imgs.transpose([3, 0, 1, 2]))

        
        return frames_tensor, torch.from_numpy(label), feature_path, nf

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    dataset_path ='/home/wluo/vidor-dataset'
    dataset = VidorDataset(dataset_path, 'training')
    dataset = VidorDataset(dataset_path, 'validation')
    


