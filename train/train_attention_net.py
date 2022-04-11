import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
from argparse import ArgumentParser
from cues import *
from torch.utils.data import Dataset
from torch.autograd import Variable
import math
from tqdm import trange
import os 
from PIL import Image
import tqdm
import argparse
from data_utils.MOT_data import get_loaders
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from spatial_attention import *

#https://github.com/hadikazemi/Machine-Learning/blob/master/PyTorch/tutorial/simese_cnn.py
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = base_lr * (0.6 ** (epoch//3)) 
    for param_group in optimizer.param_groups:
        if new_lr > 0.0000001:
            param_group['lr'] = new_lr

            
def training_loop( model, train_data_loader,optimizer,
                  val_data_loader=None,starting_epoch=1,cnn= None,device = 'cpu'  ):
    
    focal = WeightedFocalLoss()
    ce = nn.CrossEntropyLoss()
    
    min_val_loss = 100000
    new_val = False
    
    print('length of dataloader',len(train_data_loader))
    print('length of val data_laoder',len(val_data_loader))
    best_dict ={}

    for epoch in range(starting_epoch,25):
        print('in epoch',epoch)
        adjust_learning_rate(optimizer,epoch,lr)
        for p_group in optimizer.param_groups:
            print('Learning rate lowered to =',p_group['lr'])
            
        for i, sample in enumerate(train_data_loader):
            labels = sample['label'].to(device)
            if i%100 ==0:
                print('iteration:',i)
                
            new_data = sample['candidate_crop'].to(device)
            seq_data = torch.stack(sample['image_crops'])[-1].to(device)
            labels = labels.to(device)
            
            (bin_class,class_old,class_cand) = model(seq_data,new_data)

            loss_focal = focal(bin_class,labels).to(device)
            loss_ce_old  = ce(class_old,(sample["box_gt"].obj_id).long().to(device)-1).to(device)
            loss_ce_new = ce(class_cand,(sample["candidate_gt"].obj_id).long().to(device)-1).to(device)
            total_loss =  (1/2)*(loss_ce_old+loss_ce_new) + loss_focal
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            torch.cuda.empty_cache()
            
        val_loss =0
        acc =0
        total_label = 0

        for i ,val_sample in enumerate(val_data_loader):
            with torch.no_grad():
                val_label = val_sample['label'].to(device)
                new_data = val_sample['candidate_crop'].to(device)
                seq_data = (val_sample['image_crops'][-1]).to(device)
                
                val_label = val_label.to(device)

                (bin_class,class_old,class_cand) = model(seq_data,new_data)

                loss_focal = focal(bin_class,val_label).to(device)
                loss_ce_old  = ce(class_old,(val_sample["box_gt"].obj_id).to(device).long()-1).to(device)
                loss_ce_new = ce(class_cand,(val_sample["candidate_gt"].obj_id).to(device).long()-1).to(device)
                
                total_loss =  (1/2)*(loss_ce_old+loss_ce_new) + loss_focal
                
                preds = (bin_class > 0).float()
                
                acc += ((preds == val_label).sum().float()).cpu().data.numpy()
                
                
                total_label += val_label.shape[0]
                val_loss += total_loss
                
        torch.cuda.empty_cache()
        print('validation loss :',val_loss)
        print('accuracy:',acc/total_label)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            new_val =True
            print("best model will be saved into:model_saves/att_cnn"+args.cuhk+".pth.tar")
            best_dict = {'model_weight': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                 'loss': min_val_loss}
                
            torch.save(best_dict,"../model_saves/att_cnn"+args.cuhk+".pth.tar")
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Network settings
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
    parser.add_argument('--device', type=str, default='cuda',
                    help='device')
    parser.add_argument('--data_mot', type=str, default='15',
                    help='what mot data to use')
    parser.add_argument('--num_workers', type=int, default=15,
                    help='num werker')
    parser.add_argument('--cuhk', type=str, default= 'True',
                    help='add CUHK dataset for person reidentification')

    args = parser.parse_args()
    

    if args.cuhk == 'True':
        cuhk =True
    else:
        cuhk=False
    device = args.device
    train_loader, val_loader = get_loaders(
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            cuhk =cuhk,
            crops=True,
            velocities=False,
            future_seq_len=6,
            data_mot=args.data_mot,
            occupancy_grids=False,
            train_sequence_length=1,
            augment= True,
            test_sequence_length=0,
            train=True,
            test=False
    )
    
    print('RUNNING TRRAINING WITH PARAMS:')
    print('DATASET :',args.data_mot)
    print('BATCH SIZE :',args.batch_size)
    print('USING CUHK ?',args.cuhk)
    if not args.cuhk:
        print('YOU SHOULD')
    m =0
    for i,sample in enumerate(train_loader):
      #  print(sample)
        e = sample["candidate_gt"].obj_id

        m = max(torch.max(e).item(),m)
    
    print(m)
    print(type(m))
    model = SpatialAttention(is_train=True,num_classes= int(m) )
    
    model = model.to(device)
    lr =0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    starting_epoch= 0

    training_loop(model=model,train_data_loader=train_loader,optimizer=optimizer,val_data_loader = val_loader,device=device,starting_epoch=starting_epoch)
    exit(0)
