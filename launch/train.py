import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
sys.path.append("../src/")

from model import UNet
from xdataset import XDataset
from loss import DiscriminativeLoss

def train(train_path:str):
    # Model
    model = UNet().cuda()
    train_dataset = XDataset(data_path=train_path,train=True)
    train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle=False,
                                num_workers=0, pin_memory=True)
    
    criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                    delta_dist=1.5,
                                    norm=2,
                                    usegpu=True).cuda()
    criterion_ce = nn.CrossEntropyLoss().cuda()

    parameters = model.parameters()
    optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min',
                                                    factor=0.1,
                                                    patience=10,
                                                    verbose=True)

    #Trian
    model_dir = Path('../model')
    best_loss = np.inf
    for epoch in range(3000):
        print(f'epoch : {epoch}')
        disc_losses = []
        ce_losses = []
        for batched in train_dataloader:
            images, sem_labels, ins_labels = batched
            images = Variable(images).cuda()
            sem_labels = Variable(sem_labels).cuda()
            ins_labels = Variable(ins_labels).cuda()
            n_sticks = ins_labels.size()[1]
            model.zero_grad()

            sem_predict, ins_predict = model(images)
            loss = 0
            # Discriminative Loss
            disc_loss = criterion_disc(ins_predict,
                                   ins_labels,
                                   [n_sticks] * len(images))
            loss += disc_loss
            disc_losses.append(disc_loss.cpu().data.numpy())

            #Cross Entropy Loss
            _, sem_labels_ce = sem_labels.max(1)
            ce_loss = criterion_ce(sem_predict.permute(0, 2, 3, 1)\
                                    .contiguous().view(-1, 2),
                                sem_labels_ce.view(-1))
            loss += ce_loss
            #print(ce_loss.cpu().data.numpy())
            ce_losses.append(ce_loss.cpu().data.numpy())

            loss.backward()
            optimizer.step()
        disc_loss = np.mean(disc_losses)
        ce_loss = np.mean(ce_losses)
        print(f'DiscriminativeLoss: {disc_loss:.4f}')
        print(f'CrossEntropyLoss: {ce_loss:.4f}')
        scheduler.step(disc_loss)
        if disc_loss < best_loss:
            best_loss = disc_loss
            print('Best Model!')
            modelname = 'model.pth'
            torch.save(model.state_dict(), model_dir.joinpath(modelname))
    pass



if __name__ == "__main__":
    train_path = sys.argv[1]
    train(train_path=train_path)
    
