

import numpy as np
import torch
from torch.utils.data import DataLoader

import os
import numpy as np
from data_utils.data_generator import  *
from utils import get_bb_content_pil, get_velocity, get_occupancy_grid,overlap
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import cv2
import pytorch_lightning as pl
from argparse import ArgumentParser
from cues import *
from torch.utils.data import Dataset
from data_utils.data_load import *
from torch.autograd import Variable
from utils import get_bb_center
from PIL import Image
from train_svm import *
from tracker_test import *
import csv
from utils import overlap

import motmetrics
from visualize_tracker import visualize_trackers,frames_to_video


import torchreid
from utils import *
    
    
def testing_loop( detections,frames):
        tracker.reset()
        data = []
        detections_by_frame = groupby(detections,0)
        for i in range(int(np.max(list(detections_by_frame.keys())))):
            if i in detections_by_frame.keys():
                data.append(detections_by_frame[i])
            else :
                data.append([])
        dataset = dataset_dhn(data = data,frames =frames)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        assigned= 0
        for i, (dets_,frame) in enumerate(data_loader):
            if i%50 == 0:
                print('in frame i ==',i)
            
            if  type(dets_) == list:
                continue
            
            dets = dets_[0]
            img = Image.open(frame[0])
            assigned += tracker.testing_step(dets=dets,frame = img)
        
        del data_loader
        del dataset
        torch.cuda.empty_cache()

        
    
dets15,gts15, frames15 = gen_mot2015_npy()

active_svm,max_height,max_width,max_score= train_svm_active(dets15[0], gts15[0], Image.open(frames15[0][0]),0.2,0.5)

for i in range(1,len(dets15)):
    active_svm,max_height,max_width,max_score= train_svm_active(dets15[i], gts15[i], Image.open(frames15[i][0]),0.2,0.5,max_height,max_width,max_score,svm=active_svm)
    

seq_length = 6

collect_sequences = dict()
collect_start_points = list()

device = 'cuda'


H =128
k =100
grid_height = 15
grid_width= 15
subgrid_height = 7
subgrid_width = 7

d = torch.load('../model_saves/target15triplet_100_0.pth.tar')

appearance = Cue(H=500, k=500,input_fc_dim=500,is_train=False,is_app =True)    
motion = Cue(H=128, k=100,input_fc_dim=2,is_train=False,is_app =False)    
interaction = Cue(H=128, k=100,input_fc_dim=49,is_train=False,is_app =False)  


from torchreid.utils import FeatureExtractor

cnn = FeatureExtractor(
    model_name='osnet_x1_0',
   model_path='../model_saves/model.pth.tar-100',
    device='cuda'
)

target_input_dim = 700
target =  Target(H=H, k = 100,seq_len = seq_length,appearance = appearance,motion =motion,interaction=interaction,input_dim = target_input_dim,is_train=False)
target.load_state_dict(d['model_weight'])

threshold_association =0.6
threshold_overlap = 0.5
threshold_to_kill= 20
threshold_reid= 0.5
threshold_conflict= 0.7
threshold_confidence = 0.3
minimum_confidence= 0.1

target = target.to(device)
target.eval()

tracker = Tracker(target_lstm=target, cnn =cnn,active_svm=active_svm,seq_length=seq_length,
                  grid_height=grid_height, grid_width=grid_width, subgrid_height=subgrid_height,
                  subgrid_width=subgrid_width, threshold_association=threshold_association,
                  threshold_overlap=threshold_overlap,threshold_to_kill=threshold_to_kill,
                  threshold_confidence = threshold_confidence,
                  threshold_reid=threshold_reid,device=device,max_height=max_height, threshold_conflict=threshold_conflict,
                  minimum_confidence=minimum_confidence,
                  max_width=max_width,max_score=max_score)




dets15_test, frames15_test = gen_mot2015test_npy()

for i in range(len(frames15_test)):
    print(i)
    print(frames15_test[i][0])
    print(len(frames15_test[i]))


for test_idx in range(len(dets15_test)):#len(dets15)):#,1,2,4]):
    e_idx = 6
    e = frames15_test[test_idx][0].split('/')
    tracker.reset()
    print('test number ',test_idx)
    print(e[e_idx])

    testing_loop(detections = dets15_test[test_idx],frames= frames15_test[test_idx])
    write_results(targets = tracker.targets,filename='results/'+e[e_idx]+'.csv')
    tracker.reset()
    
    alltrack =[]
    with open('results/'+e[e_idx]+'.csv' ) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            alltrack.append(row)
            
    alltrack = np.array(alltrack)
    alltrack = alltrack.astype('float64')
    geoms = alltrack[:,:6]
    ids = alltrack[:,:2]

    geoms = groupby(geoms,0)
    ids = groupby(ids,0)
    for k in geoms.keys():
        geoms[k] = geoms[k][:][:,2:6]
        ids[k] = ids[k][:][:,1]
    
    print('putting in video')
    print(e[e_idx])
    from visualize_tracker import visualize_trackers,frames_to_video
    frames_to_video(frames15_test[test_idx],'videos/video'+e[e_idx]+'.mp4',fps=2)

    visualize_trackers(
        geoms,
        ids,
       'videos/video'+e[e_idx]+'.mp4',
        colormap="jet",
        alpha=1,
        output_name="videos/track"+e[e_idx]+".mp4",
        engine="matplotlib",
    )
