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
from torchvision import transforms, utils, models
from torchreid.models import build_model
from torchreid.utils import FeatureExtractor
from spatial_attention import *
import torchreid



def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = base_lr * (0.1 ** (epoch//10)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def training_loop( model, train_data_loader,optimizer,name,
                  val_data_loader=None,starting_epoch=1,cnn= None,device = 'cpu',blackout=False, ):
    
    criterion = nn.BCEWithLogitsLoss()
    min_val_loss = 10000
    new_val = False

    print('length of dataloader',len(train_data_loader))
    print('length of val data_laoder',len(val_data_loader))
    best_dict ={}

    for epoch in range(starting_epoch,50):
        print('in epoch',epoch)
        adjust_learning_rate(optimizer,epoch,lr)
        for p_group in optimizer.param_groups:
            print('Learning rate lowered to =',p_group['lr'])
       
        model.train()
        for i, sample in enumerate(train_data_loader):
            
            labels = sample['label'].to(device)
            if i%100 ==0:
                print('iteration:',i)
                
            if name == 'motion':
                seq_data = torch.stack(sample['sequence_velocities']).to(device)
                new_data = sample['candidate_velocity'].to(device)
                
            elif name == 'appearance':   
                new_data = sample['candidate_crop'].to(device)
                image_crops = sample['image_crops']
                seq_data = [] 
                with torch.no_grad():
                    for i, app_ts in enumerate(image_crops):
                        ts_f = cnn(app_ts.to(device))
                        seq_data.append(ts_f)
                    seq_data = torch.stack(seq_data).to(device)
                    new_data = cnn(new_data.to(device)).to(device)
                labels = labels.to(device)
                
            elif name =='interaction':
                seq_data  = torch.stack(sample['sequence_occupancy_grids']).to(device)
                new_data =  sample['candidate_occupancy_grid'].to(device)
                
            elif name == 'target':
                seq_app = [] 
                with torch.no_grad():
                    if use_appearance or only_appearance:
                        new_app = sample['candidate_crop'].to(device)
                        image_crops = sample['image_crops']
                        if not args.attention:
                            for i, app_ts in enumerate(image_crops):
                                ts_f = cnn(app_ts.to(device))
                                seq_app.append(ts_f)
                            seq_app = torch.stack(seq_app).to(device)
                            new_app = cnn(new_app.to(device)).to(device)
                        else:
                            seq_app = torch.stack(image_crops).to(device)
                            
                seq_int  = torch.stack(sample['sequence_occupancy_grids']).to(device)
                new_int =  sample['candidate_occupancy_grid'].to(device)
                seq_motion = torch.stack(sample['sequence_velocities']).to(device)
                new_motion = sample['candidate_velocity'].to(device)
           
            if name != 'target':
                out = model(seq_data,new_data)
            else :
                if use_appearance and not only_appearance :
                    out = model(seq_app,new_app,seq_motion,new_motion,seq_int,new_int).squeeze(1)
                elif only_appearance:
                    out = model(seq_app,new_app,None,None,None,None).squeeze(1)
                else:
                    out = model(None,None,seq_motion,new_motion,seq_int,new_int).squeeze(1)
            loss = criterion(out,labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            torch.cuda.empty_cache()
            
        val_loss =0
        acc =0
        total_label = 0
        model.eval()
        for i ,val_sample in enumerate(val_data_loader):
            with torch.no_grad():
                val_label = val_sample['label'].to(device)
                if name == 'motion':
                    seq_data = torch.stack(val_sample['sequence_velocities']).to(device)
                    new_data = val_sample['candidate_velocity'].to(device)
                elif name == 'appearance':   
                    new_data = val_sample['candidate_crop'].to(device)
                    image_crops = val_sample['image_crops']
                    seq_data = [] 
                    for  app_ts in image_crops:
                        ts_f = cnn(app_ts.to(device))
                    seq_data.append(ts_f)
                    seq_data = torch.stack(seq_data).to(device)
                    new_data = cnn(new_data).to(device)
                elif name =='interaction':
                    seq_data  = torch.stack(val_sample['sequence_occupancy_grids']).to(device)
                    new_data =  val_sample['candidate_occupancy_grid'].to(device)
                elif name == 'target':
                    seq_app = [] 
                    with torch.no_grad():
                        if use_appearance or only_appearance:
                            new_app = val_sample['candidate_crop'].to(device)
                            image_crops = val_sample['image_crops']
                            if not args.attention:
                                for i, app_ts in enumerate(image_crops):
                                    ts_f = cnn(app_ts.to(device))
                                    seq_app.append(ts_f)
                                seq_app = torch.stack(seq_app).to(device)
                                new_app = cnn(new_app.to(device)).to(device)
                            else:
                                seq_app = torch.stack(image_crops).to(device)
                    seq_int  = torch.stack(val_sample['sequence_occupancy_grids']).to(device)
                    new_int =  val_sample['candidate_occupancy_grid'].to(device)
                    seq_motion = torch.stack(val_sample['sequence_velocities']).to(device)
                    new_motion = val_sample['candidate_velocity'].to(device)
           
                if name != 'target':
                    val_out = model(seq_data,new_data)
                else :
                    if use_appearance and not only_appearance :
                        val_out = model(seq_app,new_app,seq_motion,new_motion,seq_int,new_int).squeeze(1)
                    elif only_appearance:
                        val_out = model(seq_app,new_app,None,None,None,None).squeeze(1)
                    else:
                        val_out = model(None,None,seq_motion,new_motion,seq_int,new_int).squeeze(1)
                
            
            val_loss +=  criterion(val_out,val_label)
            preds = (val_out > 0).float()
            acc += ((preds == val_label).sum().float()).cpu().data.numpy()
            total_label += val_label.shape[0]
                
            torch.cuda.empty_cache()
            
        print('accuracy:',acc/total_label)       
        print('validation loss :',val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("best model is saved ")
            best_dict = {'model_weight': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss': min_val_loss}
            if args.gt:
                if  name =='target':
                    cues = ''
                    if args.use_motion:
                        cues +='_motion'
                    if args.use_interaction:
                        cues+='_int'
                    if args.use_appearance:
                        cues+='_app'
                    torch.save(best_dict,"../model_saves/gt"+name+cues+args.cnn_loss+'_'+str(args.k)+'_'+args.test_seq+".pth.tar")
                else :
                    if name== 'appearance':
                        torch.save(best_dict,"../model_saves/gt"+name+args.cnn_loss+'_'+args.data_mot+'_'+str(args.k)+'_'+args.test_seq+".pth.tar")
                    else:
                        torch.save(best_dict,"../model_saves/gt"+name+args.data_mot+'_'+str(args.k)+'_'+args.test_seq+".pth.tar")

                
                
            else:
                addings = ''
                if args.add_negatives:
                    addings+='_negs'
                if args.add_occlusions:
                    addings+= 'occls'

                if  name =='target':
                    cues = ''
                    if args.use_motion:
                        cues +='_motion'
                    if args.use_interaction:
                        cues+='_int'
                    if args.use_appearance:
                        cues+='_app'
                        if args.attention:
                            cues +='att'
                    torch.save(best_dict,"../model_saves/"+name+addings+'_'+
                           cues+args.cnn_loss+'_'+str(args.k)+'_'+args.test_seq+".pth.tar")
                else :
                    if name== 'appearance':
                        torch.save(best_dict,"../model_saves/"+name+addings+'_'+args.cnn_loss+'_'+args.data_mot+'_'+str(args.k)+'_'+args.test_seq+".pth.tar")
                    else:
                        torch.save(best_dict,"../model_saves/"+name+addings+'_'+args.data_mot+'_'+str(args.k)+'_'+args.test_seq+".pth.tar")


    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # Network settings
    parser.add_argument('--name', type=str, default='',
                        help='which cue to train')
    parser.add_argument('--k', type=int, default=100,
                    help='k')
    parser.add_argument('--H', type=int, default=128,
                    help='H')
    parser.add_argument('--seq_len', type=int, default=6,
                    help='seq_len')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
    parser.add_argument('--device', type=str, default='cuda',
                    help='deviceee')
    parser.add_argument('--blackout', type=bool, default=False,
                    help='add blackout mask to training')
    parser.add_argument('--data_mot', type=str, default='15',
                    help='data mot to use')
    
    parser.add_argument('--test_seq', type=str, default='0',
                    help='training data for testing')
    
    parser.add_argument('--num_workers', type=int, default=10,
                    help='num werker')
    parser.add_argument('--cnn_loss', type=str, default= 'softmax',
                    help='what cnn loss was used')
    
    parser.add_argument('--use_appearance', type=str, default= 'True',
                    help='use appearance cue when training the whole network')
    parser.add_argument('--only_appearance', type=str, default= 'False',
                help='use appearance cue when training the whole network')
    parser.add_argument('--use_motion', type=str, default= 'True',
                help='use motion cue when training the whole network')
    parser.add_argument('--use_interaction', type=str, default= 'True',
            help='use interaction cue when training the whole network')
    
    parser.add_argument('--gt', type=str, default= 'False',
        help='use gt data when training')

    parser.add_argument('--attention', type=str, default= 'False',
            help='use the spatial attention for the appearance cue')
    parser.add_argument('--add_negatives', type=str, default= 'False',
            help='Use random other ids in the history')
    
    parser.add_argument('--add_occlusions', type=str, default= 'False',
            help='Put occlusions in the history')

    args = parser.parse_args()
    
    if args.use_appearance == 'True':
        use_appearance = True
    else :
        use_appearance =False
        args.use_appearance= False
    if args.only_appearance == 'True':
        only_appearance = True
        args.use_motion=False
        args.use_interaction =False
    else :
        only_appearance = False
        
    if args.use_motion == 'True':
        args.use_motion= True
    else:
        args.use_motion= False
    if args.use_interaction == 'True':
        args.use_interaction= True
    else:
        args.use_interaction= False

    if args.attention == 'True':
        args.attention=True
    else :
        args.attention = False
    
    if args.gt == 'True':
        args.gt =True
    else:
        args.gt=False
    
    if args.add_negatives ==  'True':
        args.add_negatives = True
    else:
        args.add_negatives = False
        
    if args.add_occlusions ==  'True':
        args.add_occlusions = True
    else:
        args.add_occlusions = False
    H =args.H
    k =args.k
    device = args.device
    name = args.name
    crops = False
    velocities = False
    interaction = False
    cnn = None
    torch.multiprocessing.set_sharing_strategy('file_system')
    seq_for_target =0
    seq_len = args.seq_len
    optimizer2=None
    
    
    if name =='appearance':
        is_app =True
        input_fc_dim =-1
        crops = True
        H =128
        k =100
        seq_len =6
            
        if args.cnn_loss == 'softmax':
            cnn = FeatureExtractor(
                model_name='osnet_x1_0',
               model_path='../model_saves/cnn_softmaxFalse.pth.tar',
                device='cuda'
            )
        elif args.cnn_loss == 'triplet':
            cnn = FeatureExtractor(
                model_name='osnet_x1_0',
               model_path='../model_saves/cnn_tripletFalse.pth.tar',
                device='cuda'
            )
            
        model = Cue(128,100,input_fc_dim =512,is_app=True,is_train=True)
            
        args.use_interaction =False
        args.use_motion = False

    elif name =='motion':
        is_app = False
        input_fc_dim =2
        velocities = True
        H =128
        args.use_interaction =False
        args.use_appearance = False
        model = Cue(H=H, k= args.k,input_fc_dim=input_fc_dim,is_train=True,is_app =is_app)
        
    elif name =='interaction':
        args.use_motion =False
        args.use_appearance = False
        is_app = False
        input_fc_dim =49
        interaction =True
        H =128
        model = Cue(H=H, k=args.k,input_fc_dim=input_fc_dim,is_train=True,is_app =is_app)
        
    elif name =='target' :
        addings = ''
        if args.add_negatives:
            addings+='_negs'
        if args.add_occlusions:
            addings+= '_occls'

        if use_appearance or only_appearance:
            if args.attention:
                appearance = SpatialAttention(is_train=False,num_classes=1274)

                appearance_check = torch.load('../model_saves/att_cnn.pth.tar')
                appearance.load_state_dict(appearance_check['model_weight'],strict=False)
                input_app_dim = 512
                cnn = None
                appearance.eval()
                for param in appearance.parameters():
                    param.requires_grad = False

            else:

                appearance = Cue(H=128, k=100,input_fc_dim=512,is_train=False,is_app =True)
                if args.gt:
                    app_check = torch.load('../model_saves/gtappearance'+args.cnn_loss+args.data_mot+'_'+str(k)+'_'+args.test_seq+'.pth.tar')
                else:
                    

                    app_check = torch.load('../model_saves/appearance'+addings+'_'+args.cnn_loss+'_'+args.data_mot+'_'+str(k)+'_'+args.test_seq+'.pth.tar')
                appearance.load_state_dict(app_check['model_weight'],strict=False)
                crops = True
                if args.cnn_loss == 'softmax':
                    cnn = FeatureExtractor(
                        model_name='osnet_x1_0',
                       model_path='../model_saves/cnn_softmaxFalse.pth.tar',
                        device='cuda'
                    )
                elif args.cnn_loss == 'triplet':
                    cnn = FeatureExtractor(
                        model_name='osnet_x1_0',
                       model_path='../model_saves/cnn_tripletFalse.pth.tar',
                        device='cuda'
                    )
                input_app_dim = 100
            crops =True
        else :
            
            appearance = None
            crops = False
            cnn = None
            input_app_dim =0
        
        
        if not only_appearance:
            input_dim = 0
            if args.use_interaction:
                input_dim+=100
                interaction = Cue(H=128, k=100,input_fc_dim=49,is_train=False,is_app =False)  
                if args.gt:
                    int_check  =torch.load('../model_saves/gtinteraction'+args.data_mot+'_'+str(k)+'_'+args.test_seq+'.pth.tar')
                else:
                    int_check  =torch.load('../model_saves/interaction'+addings+'_'+args.data_mot+'_'+str(k)+'_'+args.test_seq+'.pth.tar')
                interaction.load_state_dict(int_check['model_weight'])
                print('USING INTERACTION')
            else:
                interaction = None
            if args.use_motion:
                input_dim+=100
                motion = Cue(H=128, k=100,input_fc_dim=2,is_train=False,is_app =False)    
                if args.gt:
                    motion_check = torch.load('../model_saves/gtmotion'+args.data_mot+'_'+str(k)+'_'+args.test_seq+'.pth.tar')
                else:
                    motion_check = torch.load('../model_saves/motion'+addings+'_'+args.data_mot+'_'+str(k)+'_'+args.test_seq+'.pth.tar')

                motion.load_state_dict(motion_check['model_weight'])
                print('USING MOTION')

            else:
                motion = None
        
        
        if use_appearance and not only_appearance :
            target = Target(H=128, k = 100,seq_len = 6,appearance = appearance,
                            input_dim =input_app_dim +input_dim ,motion=motion,interaction=interaction,is_train=True)
        elif only_appearance :
            target = Target(H=128, k = 100,seq_len = 6,appearance = appearance,
                input_dim =input_app_dim,motion=None,interaction=None,is_train=True)
            
        elif not use_appearance :
            target = Target(H=128, k = 100,seq_len = 6,appearance = appearance,
                input_dim =input_dim,motion=motion,interaction=interaction,is_train=True)

            
        target.cuda()
        model = target
        seq_for_target = 6
        interaction = True
        velocities = True

    video_names = None
    if args.test_seq == '1':
        video_names =['TUD-Stadtmitte']
    elif args.test_seq == "2":
        video_names =['ETH-Bahnhof']
    elif args.test_seq == '3':
        video_names =['ADL-Rundle-6']
    elif args.test_seq == '4':
        video_names =['KITTI-13','ADL-Rundle-6']
    elif args.test_seq == '5':
        video_names =['TUD-Stadtmitte','KITTI-13','ETH-Bahnhof',"ADL-Rundle-6"]

    print('test seq'+args.test_seq)

    model.to(device)
    
    if args.gt:
        train_loader, val_loader = get_loaders_gt(
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                crops=crops,
                data_mot= args.data_mot,
                velocities=velocities,
                future_seq_len=6,
                occupancy_grids=interaction,
                train_sequence_length=seq_len+seq_for_target,
                test_sequence_length=seq_len+seq_for_target,
                train=True,
                test=False,
                video_names=video_names
                )
    elif args.add_negatives :
        train_loader, val_loader = get_loaders(
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            crops=crops,
            data_mot= args.data_mot,
            velocities=velocities,
            future_seq_len=seq_len,
            occupancy_grids=interaction,
            train_sequence_length=seq_len+seq_for_target,
            test_sequence_length=seq_len+seq_for_target,
            train=True,
            test=False,
            video_names=video_names,
            add_negatives=True
            )
        
    elif args.add_occlusions :
        train_loader, val_loader = get_loaders(
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            crops=crops,
            data_mot= args.data_mot,
            velocities=velocities,
            future_seq_len=seq_len,
            occupancy_grids=interaction,
            train_sequence_length=seq_len+seq_for_target,
            test_sequence_length=seq_len+seq_for_target,
            train=True,
            test=False,
            video_names=video_names,
            add_occlusions=True
            )

    
    
    else:
        train_loader, val_loader = get_loaders(
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        crops=crops,
        data_mot= args.data_mot,
        velocities=velocities,
        future_seq_len=seq_len,
        occupancy_grids=interaction,
        train_sequence_length=seq_len+seq_for_target,
        test_sequence_length=seq_len+seq_for_target,
        train=True,
        test=False,
        video_names=video_names
        )
        
        

    lr =0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    starting_epoch= 1
    print('RUNNING TRRAINING WITH PARAMS:')
    print('NAME OF MODEL:',args.name)
    print('DATASET :',args.data_mot)
    print('BATCH SIZE :',args.batch_size)
    print('CNN TRAINED WITH:',args.cnn_loss)
    print('USING APPEARANCE CUE:',args.use_appearance)
    training_loop(model=model,cnn= cnn,train_data_loader=train_loader,optimizer=optimizer,name=name,val_data_loader = val_loader,device=device,starting_epoch=starting_epoch, blackout=args.blackout)
    
    exit(0)
