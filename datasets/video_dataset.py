# -- coding: utf-8 --**
# the dataset class for EV-Gait-3DGraph model


import os
import numpy as np
import glob
import pdb
import scipy.io as sio
import torch
import torch.utils.data
from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
import cv2
import random

# import voxel
# import geome
# import tri


def files_exist(files):
    return all([osp.exists(f) for f in files])
def load_video(root,mode,split):
    if split == 'txt':
        root_path = os.path.join(root,'rawframes')
        labels = []
        rgb_samples = []
        anno_file = os.path.join(root,'Ncal_{}.txt'.format(mode))
        with open(anno_file, 'r') as fin:
            for line in fin:
            
                line_split = line.strip().split()
                idx = 0
                frame_dir = line_split[idx]
                temp_dir = os.path.split(frame_dir)[-1]+'_dvs'
                img_list = os.listdir(os.path.join(root_path,frame_dir))
                img_path = []
                for img in img_list:
                    
                    img_path.append(os.path.join(root_path,frame_dir,img))
                    img_path.sort()
            
                rgb_samples.append(img_path) 
                label = line_split[idx+2]
                labels.append(label)
    else:
        root_path = os.path.join(root,mode,'rawframes')
        labels = []
        rgb_samples = []
        cls_list = os.listdir(root_path)
        for cls_id in range(len(cls_list)):
            cls = cls_list[cls_id]
            video_list = os.listdir(os.path.join(root_path,cls))
            for video_id in range(len(video_list)):
                video = video_list[video_id]
                img_list = os.listdir(os.path.join(root_path,cls,video))
                if len(img_list)==0:
                    print((os.path.join(root_path,cls,video)))
                img_path = []
                for img in img_list:
                    img_path.append(os.path.join(root_path,cls,video,img))
                    img_path.sort()
                rgb_samples.append(img_path) 
                label = int(cls)
                labels.append(label)
    return rgb_samples, labels

class video_Dataset(Dataset):
    def __init__(self, root_path, mode, split=None,spatial_transform=None, temporal_transform=None):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_video(root_path, mode, split)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        #pdb.set_trace()       # 打断点做可视化1  
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            # print(rgb_name[frame_name_i])
            ori_img = Image.open(rgb_name[frame_name_i])
            rgb_cache = ori_img.convert("RGB")
            clip_frames.append(rgb_cache)

        clip_frames = self.spatial_transform(clip_frames)
        n, h, w = clip_frames.size()
        return clip_frames.view(-1, 3, h, w),   int(label)
    def __len__(self):
        return int(self.sample_num)
        

    @property
    def raw_file_names(self):
        pass

    # get all file names in  self.processed_dir
    @property
    def processed_file_names(self):
        pass
    def _process(self):
        pass
    def _download(self):
        pass

    def download(self):
        pass
    def process(self):
        pass
    def get():
        pass