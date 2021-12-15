import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy import ndimage as ndi

import sys
sys.path.append("../src/")

from model import UNet
from xdataset import XDataset
from utils import gen_color_img
import matplotlib.pyplot as plt



def predict(train_path:str):
    # load Model
    model = UNet().cuda()
    model.eval()

    model_dir = Path('../model')
    model_path = model_dir.joinpath('model.pth')

    param = torch.load(model_path)
    model.load_state_dict(param)

    predict_dataset = XDataset(data_path=train_path,train=True)
    predict_dataloader = DataLoader(predict_dataset,batch_size=1,shuffle=False,
                                num_workers=0, pin_memory=True)

    #Predict
    model_dir = Path('../model')
    pred_test=[]
    for i, (imgs, lables, ins_gt) in enumerate(predict_dataloader):
        imgs = imgs.cuda()
        lables = lables.cuda()
        print(lables.shape)
        print(imgs.shape)
        sem_pred, ins_pred= model(imgs)
        ins_pred = ins_pred.cpu().data.numpy()
        #ins_pred =np.squeeze(ins_pred, axis=0)
        print(ins_pred.shape)
        #pred = np.squeeze(sem_pred.data.max(1)[1].cpu().numpy(), axis=0)
        pred = F.softmax(sem_pred, dim=1).cpu().data.numpy()
        pred_test.append(pred)
        pred_test = np.concatenate(pred_test)[:, 1, :, :]
        print(pred_test.shape)
        gt = np.squeeze(lables.data.max(1)[1].cpu().numpy(), axis=0)
        p_sem_pred = []
        for sp in pred_test:
            p_sem_pred.append(ndi.morphology.binary_fill_holes(sp > 0.5))
        color_img = gen_color_img(p_sem_pred[0], ins_pred[0], 3)
        plt.subplot(1,3,1)
        #plt.imshow(pred)
        plt.subplot(1,3,2)
        plt.imshow(color_img)
        plt.subplot(1,3,3)
        plt.imshow(gt)
        plt.show()
    

    pass



if __name__ == "__main__":
    train_path = sys.argv[1]
    predict(train_path=train_path)