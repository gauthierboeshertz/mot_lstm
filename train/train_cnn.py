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
from data_utils.MOT_data_gt import get_loaders as get_loaders_gt

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchreid

#https://github.com/hadikazemi/Machine-Learning/blob/master/PyTorch/tutorial/simese_cnn.py
    
class siamese_cnn(torch.nn.Module):

    def __init__(self,crit):
        super(siamese_cnn, self).__init__()
        
        self.crit = crit
            #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
       # self.cnn.classifier[-1] = nn.Linear(in_features=self.cnn.classifier[1].in_features, out_features=500)
        self.cnn = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=20,
            loss='softmax',
            pretrained=True
        )
        self.cnn.classifier= nn.Linear(in_features=self.cnn.classifier.in_features, out_features=500)
        
        for param in self.cnn.parameters():
            param.requires_grad = True
        if self.crit != 'contr':
            self.train_layer = nn.Linear(in_features= 2*500, out_features=1)

    def forward(self,sequence_crop, cand_crop):
        out_cnn_seq = self.cnn(sequence_crop)
        out_cnn_cand = self.cnn(cand_crop)
        if self.crit=='contr':
            return (out_cnn_seq,out_cnn_cand)
        else :
            input_to_train = torch.cat((out_cnn_seq, out_cnn_cand),dim=-1)
            out = self.train_layer(input_to_train)

            return out

def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = base_lr * (0.3 ** (epoch//3)) 
    for param_group in optimizer.param_groups:
        if new_lr > 0.0000001:
            param_group['lr'] = new_lr

            
def training_loop( model, train_data_loader,optimizer,crit,
                  val_data_loader=None,starting_epoch=1,cnn= None,device = 'cpu'  ):
    if crit!= 'contr':
        criterion = nn.BCEWithLogitsLoss()
    
    min_val_loss = 10000
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
            if i%400 ==0:
                print('iteration:',i)
            new_data = sample['candidate_crop'].to(device)
            seq_data = torch.stack(sample['image_crops'])[0].to(device)
            labels = labels.to(device)
            
            if crit == 'contr':
                labels = 1 -labels
                (out_cnn_seq,out_cnn_cand) = model(seq_data,new_data)
                euclidean_distance = F.pairwise_distance(out_cnn_seq, out_cnn_cand)
                loss= torch.mean((1/2)*(1 - labels) * torch.pow(euclidean_distance, 2) +
                                      (1/2)*labels * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))

            else :
                out  = model(seq_data,new_data)
                loss = criterion(out.squeeze(1),labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        torch.cuda.empty_cache()
            
        val_loss =0
        acc =0
        total_label = 0

        for i ,val_sample in enumerate(val_data_loader):
            with torch.no_grad():
                val_label = val_sample['label'].to(device)
                new_data = val_sample['candidate_crop'].to(device)
                seq_data = torch.stack(val_sample['image_crops'])[0].to(device)
                
                if crit =='contr':
                    val_label = 1-val_label
                    (out_cnn_seq,out_cnn_cand) = model(seq_data,new_data)
                    euclidean_distance = F.pairwise_distance(out_cnn_seq, out_cnn_cand)
                    val_loss += torch.mean((1/2)*(1 - val_label) * torch.pow(euclidean_distance, 2) +
                                      (1/2)*val_label * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))
                else :
                    out  = model(seq_data,new_data).squeeze(1)
                    val_loss += criterion(out,val_label)
                    preds = (out > 0).float()
                    acc += ((preds == val_label).sum().float()).cpu().data.numpy()
                    total_label += val_label.shape[0]

        torch.cuda.empty_cache()
        print('validation loss :',val_loss)
        if crit != 'contr':
            print('accuracy:',acc/total_label)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            new_val =True
            print("best model will be saved into:model_saves/cnn"+crit+str(args.gt)+".pth.tar")
            best_dict = {'model_weight': model.cnn.state_dict(),
                'optimizer':  optimizer.state_dict(),
                 'loss': min_val_loss}
                
            torch.save(best_dict,"../model_saves/cnn"+crit+str(args.gt)+".pth.tar")
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Network settings
    parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
    parser.add_argument('--device', type=str, default='cuda',
                    help='deviceee')
    parser.add_argument('--data_mot', type=str, default='15_16',
                    help='whaat mot daa to use')
    parser.add_argument('--num_workers', type=int, default=15,
                    help='num werker')
    parser.add_argument('--crit', type=str, default= 'bce',
                    help='what loss to use contrastive or bce')
    parser.add_argument('--cuhk', type=bool, default= True,
                    help='add CUHK dataset for person reidentification')
    
    parser.add_argument('--gt', type=str, default= 'True',
                    help=' use gt and modify the boxes')
    args = parser.parse_args()
    
    if args.gt == 'True':
        args.gt = True
    else:
        args.gt = False
    


    device = args.device
    model = siamese_cnn(args.crit) 
    
    model = model.to(device)
    lr =0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    starting_epoch= 0
    if args.gt:
        train_loader, val_loader = get_loaders_gt(
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                cuhk =False,
                crops=True,
                velocities=False,
                future_seq_len=1,
                data_mot=args.data_mot,
                occupancy_grids=False,
                train_sequence_length=1,
                augment= True,
                test_sequence_length=0,
                train=True,
                test=False
        )
    else:
        train_loader, val_loader = get_loaders(
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                cuhk =args.cuhk,
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
    print('contrastive loss?',args.crit)
    print('USING CUHK ?',args.cuhk)
    if not args.cuhk:
        print('YOU SHOULD')
        
    training_loop(model=model,train_data_loader=train_loader,optimizer=optimizer,val_data_loader = val_loader,device=device,starting_epoch=starting_epoch,crit = args.crit)
    exit(0)
