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
import copy
from torchvision import transforms, utils, models

from data_utils.MOT_cropped import *

import numpy as np
import torch
from torch.utils.data import DataLoader

import torchreid



torchreid.data.register_image_dataset('MOT_cropped0', MOT_cropped0)
torchreid.data.register_image_dataset('MOT_cropped1', MOT_cropped1)
torchreid.data.register_image_dataset('MOT_cropped2', MOT_cropped2)
torchreid.data.register_image_dataset('MOT_cropped3', MOT_cropped3)
torchreid.data.register_image_dataset('MOT_cropped4', MOT_cropped4)
torchreid.data.register_image_dataset('MOT_cropped5', MOT_cropped5)
torchreid.data.register_image_dataset('MOT_cropped6', MOT_cropped6)
torchreid.data.register_image_dataset('MOT_cropped7', MOT_cropped7)
torchreid.data.register_image_dataset('MOT_cropped8', MOT_cropped8)
torchreid.data.register_image_dataset('MOT_cropped9', MOT_cropped9)
torchreid.data.register_image_dataset('MOT_cropped10', MOT_cropped10)
torchreid.data.register_image_dataset('MOT_cropped11', MOT_cropped11)
torchreid.data.register_image_dataset('MOT_cropped12', MOT_cropped12)
torchreid.data.register_image_dataset('MOT_cropped13', MOT_cropped13)

datamanager = torchreid.data.ImageDataManager(
    root='/media/data/gauthier/',
    sources=['MOT_cropped0','MOT_cropped1',
             'MOT_cropped2','MOT_cropped3',
             'MOT_cropped4','MOT_cropped4',
             'MOT_cropped5','MOT_cropped6',
             'MOT_cropped8','MOT_cropped7',
             'MOT_cropped9','MOT_cropped10',
             'MOT_cropped11','MOT_cropped12',
             'MOT_cropped13','cuhk03'],#,'MOT_cropped'],
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=64,
    transforms=['random_flip', 'random_crop'],
    combineall=True,
    cuhk03_labeled=True,
    train_sampler='RandomIdentitySampler'
)

print(datamanager.num_train_pids)
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='triplet',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0006

)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=8,
    gamma = 0.2,
)
"""
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True

)
"""

engine = torchreid.engine.ImageTripletEngine(
    datamanager, model, optimizer, margin=0.3,
    weight_t=0.7, weight_x=1, scheduler=scheduler
)

engine.run(
    save_dir='model/triplet/',
    max_epoch=100,
    eval_freq=10,
    print_freq=50,
    test_only=False
)
