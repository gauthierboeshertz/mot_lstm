import os
import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
import copy
from efficientnet_pytorch import EfficientNet
from spatial_attention import *


class Cue(torch.nn.Module):
    def __init__(self, H, k,input_fc_dim,is_app=False,is_train=False):
        super(Cue, self).__init__()
        self.H = H
        self.k = k
        self.is_train = is_train
        if is_app:
            self.last_fc = nn.Linear(input_fc_dim+H, self.k)
            self.lstm = nn.LSTM(input_size=input_fc_dim, hidden_size=H)

        else :
            self.lstm = nn.LSTM(input_size=input_fc_dim, hidden_size=H)
            self.last_fc = nn.Linear(2*H, self.k)
        

        self.training_layer =  nn.Linear(in_features=self.k, out_features=1)
        self.is_app = is_app
        if not is_app:
            self.first_fc = nn.Linear(input_fc_dim, self.H)

    def forward(self, feature_seq, new_feature):
        ## lstm output is  output , hidden states, c
        if type(feature_seq) == list:
            feature_seq = torch.stack(feature_seq)
        if len(new_feature.shape ) !=2:
            new_feature = new_feature.unsqueeze(0)
            
        if len(feature_seq.shape) != 3:
            feature_seq = feature_seq.unsqueeze(1)
            
        if not self.is_app:
            first_fc_out = F.relu(self.first_fc(new_feature))
        else:
            first_fc_out = new_feature
            
        out_lstm, _ = self.lstm(feature_seq)
        input_to_fc = torch.cat((out_lstm[-1], first_fc_out),dim=-1)

        out = F.relu(self.last_fc(input_to_fc))
        if  self.is_train :
            out = self.training_layer(out).view(-1)

        return out

    
class Target(torch.nn.Module):

    def __init__(self, H, k, seq_len, appearance,motion,interaction,input_dim =700,is_train=False):
        super(Target, self).__init__()
        
        self.appearance = appearance
        self.motion = motion
        self.interaction= interaction
        self.seq_len = seq_len
        self.H = H
        self.k =k
        self.top_lstm = nn.LSTM(input_size=input_dim, hidden_size=H)
        self.fc_layer = nn.Linear(H, k)
        self.output_layer = nn.Linear(k, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.is_train = is_train
    def forward(self, bb_seq, new_bb, velocity_seq, new_velocity, occ_grid_seq, new_occ_grid):
        input_to_lstm = []

        if self.appearance is not None:
            if type(self.appearance) == SpatialAttention:
                seq_att  = []
                num_bb = len(bb_seq)
                if num_bb < 6:
                    temp_seq_bb = copy.deepcopy(bb_seq)
                    tmp_list = bb_seq[::-1]
                    while len(temp_seq_bb) < 6:
                        temp_seq_bb += tmp_list
                    seq_att = temp_seq_bb[0:6]

                else:
                    gap = int(num_bb / 6)
                    mod = num_bb % 6
                    tmp_list = bb_seq
                    seq_att = []
                    for i in range(mod, num_bb, gap):
                        seq_att.append(bb_seq[i])


        for i in range(self.seq_len):
            bb_subseq= []
            velocity_subseq= []
            interaction_subseq= []
            if self.appearance is not None:
                if  type(self.appearance) == SpatialAttention:
                    bb_subseq = seq_att[i]
                else:
                    bb_subseq =list(itertools.islice(bb_seq,i , i+self.seq_len)) 
                    
            if self.motion is not None:
                velocity_subseq = list(itertools.islice(velocity_seq,i , i+self.seq_len))
            if self.interaction is not None:
                interaction_subseq = list(itertools.islice(occ_grid_seq,i , i+self.seq_len))
            
            if len(velocity_subseq) or len(bb_subseq)  or len(interaction_subseq):
                subnet_out = []
                
                if self.motion is not None:
                    motion_f = self.motion(velocity_subseq, new_velocity)
                    subnet_out.append(motion_f)
                    
                if self.interaction is not None:
                    interaction_f = self.interaction(interaction_subseq, new_occ_grid)
                    subnet_out.append(interaction_f)
                    
                if self.appearance is not None:
                    app_f = self.appearance(bb_subseq, new_bb)
                    subnet_out.append(app_f)

                input_to_lstm.append(torch.cat(subnet_out,dim=1))
               
                
            else:
                break
        if len(input_to_lstm):
            input_to_lstm = torch.stack(input_to_lstm)
            out_lstm, (hn, cn) = self.top_lstm(input_to_lstm)
            input_to_last_fc = F.relu(self.fc_layer(out_lstm[-1]))
            out = self.output_layer(input_to_last_fc)
            if not self.is_train:
                out = self.sigmoid(out)
        else :
            raise Exception('sequences are empty for top lstm')
        return out
