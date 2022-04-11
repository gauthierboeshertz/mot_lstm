from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pprint
import time
import itertools
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import copy
import shutil
from tensorboardX import SummaryWriter
import sys
import random
import os
import numpy as np
from data_utils.data_generator import  *
from utils import get_bb_content_pil, get_velocity, get_occupancy_grid,overlap
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Normalize, Compose, ToTensor
import cv2
import pytorch_lightning as pl
from argparse import ArgumentParser
from cues import *
from torch.utils.data import Dataset
from data_utils.data_load import *
from torch.autograd import Variable
import math
from utils import get_bb_center
from PIL import Image
from tracker_train import Tracker
from torchreid.models import build_model
from torchreid.utils import FeatureExtractor

from dhn.DHN import Munkrs

def groupby(x): 
    x_uniques = np.unique(x[:,0]) 
    return {xi:x[x[:,0]==xi] for xi in x_uniques} 


def adjust_learning_rate(optimizer, epoch, base_lr):
    new_lr = base_lr * (0.8 ** (epoch)) 
    for param_group in optimizer.param_groups:
        if new_lr > 0.0000001:
            param_group['lr'] = new_lr




def training_data(seq_length = 6,split_sequence = 15,min_seq_length  = 5,gts_list = []):
    collect_sequences = dict()
    collect_start_points = list()
    for i,gts in enumerate(gts_list):
        # random start points #
        gt_grouped_by_frame = groupby(gts)
        start_points = list()  # index i to start during training
        seq_len = len(gt_grouped_by_frame)
        start_points.append(str(i) + '_1')
        num_small_seqs = int((seq_len - 1) // split_sequence)  # start with zero
        if seq_len > split_sequence:
            for j in range(1,num_small_seqs + 1):
                if j* split_sequence in gt_grouped_by_frame.keys():
                    start_points.append(str(i) + '_' + str(j * split_sequence))
            start_points.pop(-1)
            start_points = start_points  # every 10 frames
            collect_sequences[i] = gt_grouped_by_frame
            np.random.shuffle(start_points)
            collect_start_points += start_points
    return collect_sequences,collect_start_points
    
def training_loop(epoch= 20,collect_start_points = dict(),
                  collect_start_points_val= dict(),
                  collect_sequences =dict(),
                  split_sequence = 16,
                 starting_epoch=1):
    min_val_loss = 1000000
    loss_to_plot = []
    time_lr_adjusted = 0
    for epoch in range(starting_epoch, epoch):
        np.random.shuffle(collect_start_points)
        for iteration,start_pt in enumerate(collect_start_points):
            if iteration != 0 and  (iteration%(len(collect_start_points)//3))==0:
                adjust_learning_rate(optimizer,time_lr_adjusted,lr)
                time_lr_adjusted +=1
                for p_group in optimizer.param_groups:
                    print('Learning rate lowered to =',p_group['lr'])
            seq_number = int(start_pt.split('_')[0])
            start_point = int(start_pt.split('_')[-1])
            data = []
            frames = []
            for i in range(split_sequence):
                if start_point+i in collect_sequences[seq_number]:
                    data.append( collect_sequences[seq_number][start_point+i])
                    frames.append(frames_list[seq_number][start_point-1+i])
            dataset = dataset_dhn(data = data,frames =frames)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            loss =0
            
            tracker.reset()

            for i, (gts,frame) in enumerate(data_loader):
                img = Image.open(frame[0])
                gts_drop  = gts[0:,:20]
                    
                if i <=1:
                    loss += tracker.training_step(gts=gts_drop,frame = img, epoch = epoch,
                                                 i=i,mix_asso = False )
                else:
                    loss += tracker.training_step(gts=gts_drop,frame = img, epoch = epoch,
                                                  i=i,
                                                  mix_asso =  0 if epoch < 2 else iteration%2 ==0,
                                                  dropping_frames = iteration%3==0 )
                    
            print('continuing iteration',iteration,'loss is ',loss)
            del dataset
            del data_loader
            torch.cuda.empty_cache()
            validation_loss = 0
            
                
            if (( iteration%((len(collect_start_points)//3))==0 or iteration==len(collect_start_points)-2) and iteration !=0) :
                print(' ! validating ! ')
                for start_pt_val in collect_start_points_val :
                    seq_number = int(start_pt_val.split('_')[0])
                    start_point = int(start_pt_val.split('_')[-1])
                    data = []
                    frames = []
                    for i in range(split_sequence):
                        if start_point+i in collect_sequences[seq_number]:
                            data.append( collect_sequences[seq_number][start_point+i])
                            frames.append(frames_list[seq_number][start_point-1+i])
                    dataset = dataset_dhn(data = data,frames =frames)
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
                    
                    tracker.reset()

                    for i, (gts,frame) in enumerate(data_loader):
                        gts_drop  = gts[0:,:20]
                        img = Image.open(frame[0])
                        validation_loss += tracker.validation_step(gts=gts_drop,frame = img, epoch = epoch,i=i)
                        
                    torch.cuda.empty_cache()
                    del dataset
                    del data_loader
                print('Validation loss=',validation_loss)
                print('Min validation loss = ',min_val_loss)
                loss_to_plot.append([epoch,validation_loss])
                if validation_loss < min_val_loss:
                    min_val_loss = validation_loss
                    print("best model is saved into: ../model_saves/nl_MOT17_best_model_" + str(epoch) + ".pth.tar")
                    torch.save({'model_weight': tracker.target_lstm.state_dict(),
                        'optimizer':  tracker.optimizer.state_dict(),
                           'k':tracker.target_lstm.k,
                           'H':tracker.target_lstm.H,
                        loss: min_val_loss},
                        "../model_saves/nl_MOT15_target" + str(epoch) + ".pth.tar")


                    
                    
detections_list,gts_list, frames_list = gen_mot2016_npy()

for i in range(len(gts_list)):
    gts = gts_list[i]
    gts_bad = gts[:,6] <0.5
    gts_ = np.delete(gts,gts_bad,0)
    gts_list[i] = gts_
    
    
    

seq_length=6
split_sequence = 13

H = 128
k = 100
grid_height= 15
grid_width= 15
subgrid_height = 7
subgrid_width = 7

appearance = Cue(H=500, k=500,input_fc_dim=500,is_train=False,is_app =True)    
motion = Cue(H=H, k=k,input_fc_dim=2,is_train=False,is_app =False)    
interaction = Cue(H=H, k=k,input_fc_dim=49,is_train=False,is_app =False)  
        

starting_epoch=1

    
interaction = interaction.to('cuda')
motion = motion.to('cuda')
appearance = appearance.to('cuda')


target = Target(H=128, k = 100,seq_len = 6,input_dim = 500+2*k,
                appearance = appearance,motion=motion,interaction=interaction,is_train=False)

target.cuda()


lr =0.0005
optimizer = torch.optim.Adam(target.parameters(), lr=lr)

threshold_association =0.5
threshold_overlap = 0.3
threshold_to_kill= 10

device = 'cuda'

cnn = FeatureExtractor(
    model_name='osnet_x1_0',
   model_path='../model_saves/cnn_triplet.pth.tar',
    device='cuda'
)

dhn = Munkrs(element_dim=1, hidden_dim=256, target_size=1,
                 bidirectional=True, minibatch=1, is_cuda=True,
                 is_train=False)


dhn.cuda()

dhn.load_state_dict(torch.load('dhn/DHN.pth',map_location=torch.device('cuda')))

for param in dhn.parameters():
    param.requires_grad = False

tracker = Tracker(target_lstm=target, dhn=dhn,cnn=cnn,active_svm=None, optimizer=optimizer,seq_length=seq_length, grid_height=grid_height, grid_width=grid_width, subgrid_height=subgrid_height, subgrid_width=subgrid_width, threshold_association=threshold_association, threshold_overlap=threshold_overlap,threshold_to_kill=threshold_to_kill, threshold_reid = 0.2,device=device)

     
collect_sequences,collect_start_points = training_data(seq_length=seq_length,
                                                       split_sequence = split_sequence,
                                                       min_seq_length  = 8,
                                                       gts_list = gts_list)


random.shuffle(collect_start_points)
collect_start_points_train = collect_start_points[: int(9*len(collect_start_points)/10)+1]
collect_start_points_val = collect_start_points[ int(9*len(collect_start_points)/10):]


starting_epoch =1

for param in target.parameters():
    param.requires_grad = True

training_loop(epoch= 20,collect_start_points=collect_start_points_train,
              collect_start_points_val  = collect_start_points_val,
              collect_sequences=collect_sequences,
              split_sequence=split_sequence,starting_epoch=starting_epoch)
