import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

import pdb

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
# load data
# load images from the directory
# load segmentation from the directory

class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, self.mode, 'img')
        self.seg_dir = os.path.join(self.data_dir, self.mode, 'seg')

        # make img_data array to read the images
        mypath = os.listdir(self.img_dir)
        # print(mypath.__len__())
        self.img_data = []
        for i in range(0,mypath.__len__()):
            # print(i)
            temp = '/'.join([self.img_dir, mypath[i]])
            self.img_data.append(temp)
        
        # make seg_data array to read the images
        mypath2 = os.listdir(self.seg_dir)
        # print(mypath2.__len__())
        self.seg_data = []
        for i in range(0,mypath2.__len__()):
            # print(i)
            temp = '/'.join([self.seg_dir, mypath[i]])
            self.seg_data.append(temp)
        
        # pdb.set_trace()

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])


    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img_data[idx]
        seg_path = self.seg_data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path).convert('RGB')
        # pdb.set_trace()
        return self.transform(img), self.transform(seg)
