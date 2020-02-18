import os
import argparse
import pickle
from dataset import VidOR
from tqdm import tqdm
import glob

dataset_path ='/home/wluo/vidor-dataset'
anno_path = os.path.join(dataset_path,'annotation')
video_path = os.path.join(dataset_path,'video')
frame_path = os.path.join(dataset_path,'frame')
local_ffmpeg_path = '/home/wluo/ffmpeg-3.3.4/bin-linux/ffmpeg'

def load_vidor_dataset():
    if not os.path.exists('dataset/vidor-dataset.pkl'):
        dataset = VidOR(anno_path, video_path, ['training', 'validation'], low_memory=True)
        with open('dataset/vidor-dataset.pkl','wb') as file:
            pickle.dump(dataset,file)
    else:
        with open('dataset/vidor-dataset.pkl','rb') as file:
            dataset = pickle.load(file)
    return dataset


def extract_frames(split):
    assert split in ['training', 'validation']
    dataset = load_vidor_dataset()
    vids = dataset.get_index(split)
    video_paths = [dataset.get_video_path(vid,split) for vid in vids]
    for video_path in tqdm(video_paths):
        frame_path = video_path.replace('video','frame').replace('.mp4','')
        if not os.path.exists(frame_path):
            os.makedirs(frame_path, exist_ok=True)
            os.system(local_ffmpeg_path + ' -i ' + video_path + ' ' + frame_path + '/%4d.jpg > /dev/null 2>&1')
        frames = sorted(glob.glob(frame_path+'/*.jpg'))
        assert len(frames)>0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, default="validation")
    args = parser.parse_args()
    extract_frames(args.split)