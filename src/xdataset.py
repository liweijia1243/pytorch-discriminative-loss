import numpy as np
import cv2
import os

import torch
from torch.utils.data import Dataset, dataset
from PIL import Image
from PIL import ImageFile


class XDataset(Dataset):
    def __init__(self, data_path : str,train : bool, img_size=512):
        super().__init__()
        self.data_path = data_path
        self.label_files = []
        with open(data_path, "r") as file:
            self.img_files = file.readlines()

        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images",1))
            label_file = os.path.join(label_dir, os.path.basename(path))
            self.label_files.append(label_file)

        self.train = train
        self.height = img_size
        self.width = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        import matplotlib.pyplot as plt
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        #print(img_path)
        img = np.array(Image.open(img_path).convert('L'),dtype=np.uint8)
        #n_sticks = int(img_path.split('/')[-1].split('_')[1].split('.')[0])
        #print(n_sticks)

        # ---------
        #  Label
        # ---------
        ins = np.zeros((0, self.height, self.width), dtype=np.uint8)
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        img_label = np.array(Image.open(label_path).convert('L'),dtype=np.uint8)
        instance=np.unique(img_label)
        n_sticks = len(instance)-1
        for value in instance:
            if value!=0:
                gt = np.zeros_like(img)
                gt[img_label ==value] = 1
                ins = np.concatenate([ins,gt[np.newaxis]])
        #print(len(ins))
        
        if self.train:
            sem = np.zeros_like(img_label, dtype=bool)
            sem[img_label != 0] = True
            sem = np.stack([~sem, sem]).astype(np.uint8)

            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            # 2 * height * width
            sem = torch.Tensor(sem)
            # n_sticks * height * width
            ins = torch.Tensor(ins)
            return img, sem, ins
        else:
            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            return img

        pass

if __name__=="__main__":
    xdataset = XDataset("../datasets/train.txt", train=True, img_size=512)
    print(xdataset.img_files)
    xdataset.__getitem__(100)