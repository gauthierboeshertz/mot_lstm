
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
from utils import overlap,get_velocity,get_bb_content_pil,get_bb_content_pil_opened,get_occupancy_grid
import torch
from dhn.DHN import *
from dhn.loss_utils import  *
import copy
import random

class Tracker:
    def __init__(self, target_lstm=None, dhn=None,cnn =None, active_svm=None, optimizer=None ,batch_size =1, seq_length=6, grid_height=15, grid_width=15, subgrid_height=7, subgrid_width=7, threshold_association=0.8, 
                 threshold_overlap=0.3,threshold_to_kill=50,threshold_reid=0.3,device = 'cuda'):
        
        self.targets = []
        self.active_targets = []
        self.target_lstm = target_lstm
        self.device = device
        self.dhn = dhn
        self.cnn = cnn
        self.optimizer = optimizer
        self.active_svm = active_svm
        self.seq_length = seq_length
        self.prev_asso = {} 
        self.batch_size = batch_size
        self.threshold_association = threshold_association
        self.threshold_overlap = threshold_overlap
        self.threshold_to_kill = threshold_to_kill
        self.threshold_reid = threshold_reid
        self.grid_height= grid_height
        self.grid_width = grid_width
        self.subgrid_height = subgrid_height
        self.subgrid_width = subgrid_width
    '''
    def track(self, video_seq, detections):
    '''
    def reset(self):
        for t in self.targets:
            del t
        self.targets = []
        self.prev_asso = {}
        self.active_targets = []
        torch.cuda.empty_cache()
        
    def training_step(self,gts,frame,epoch,i,mix_asso =False,dropping_frames =False):
        if i ==0 or len(self.active_targets) ==0:
            if gts.shape[0] > 0:
                self.sort_to_birth_train(gts=gts[0], gts_to_birth=gts[0], frame=frame)
                self.prev_asso = self.update_asso(self.prev_asso)
                return 0

        else:
            to_ret = -1
            if gts.shape[1] > 0:
                if(len(self.active_targets)):
                    # if we have targets track them
                    ##for all targets compute their distance to the gt detections
                        
                                                
                    similarity_gt = []
                    gts_ints = []
                    input_to_cnn = []
                    for gt in gts[0]:
                        gts_ints += [get_occupancy_grid(detection=gt, detection_list=gts[0],
                                         frame_width= frame.size[0],frame_height=frame.size[1],
                                         grid_height=self.grid_height,grid_width=self.grid_width,
                                         subgrid_height=self.subgrid_height,
                                        subgrid_width=self.subgrid_width).to(self.device)]

                        
                        input_to_cnn += [get_bb_content_pil_opened(frame=frame, detection=gt).to(self.device)]
                        
                    with torch.no_grad():
                        gts_apps = self.cnn(torch.stack(input_to_cnn)).unsqueeze(1).to(self.device)

                    for i,target in enumerate(self.active_targets):
                        similarity_gt_for_t = torch.zeros(( gts.shape[1],))
                        for idx in range(gts.shape[1]):
                            new_motion = get_velocity(det_t_minus_1=target.tracks[-1],
                                                      det_t=gts[0][idx]).to(self.device)
                            similarity_gt_for_t[idx]= self.target_lstm(bb_seq = target.apps,new_bb= gts_apps[idx],
                                                                     velocity_seq=target.motions,new_velocity= new_motion,
                                                                     occ_grid_seq=target.interactions,new_occ_grid= gts_ints[idx]
                                                                        )
                               
                        similarity_gt.append(similarity_gt_for_t)
                    ## we want distance and in the shape (tracker, detections):
                    similarity_gt = torch.stack(similarity_gt)


                    distance_gt = 1 - similarity_gt
                    distance_gt = distance_gt.unsqueeze(0).to(self.device).contiguous()
                    output_dhn = self.dhn(distance_gt).to(self.device)
                    softmaxed_row = rowSoftMax(output_dhn, scale=100.0, threshold=0.5).contiguous()
                    softmaxed_col = colSoftMax(output_dhn, scale=100.0, threshold=0.5).contiguous()

                    fn = missedObjectPerframe(softmaxed_col)
                    fp = falsePositivePerFrame(softmaxed_row)

                    hypo_ids = [t.id for t in self.active_targets]
                    gt_ids = [int(gt_id) for gt_id in gts[0,:,1]]
                    mm,  motp_mask, self.prev_asso,self.active_targets = missedMatchErrorV3_tracktor(prev_id=self.prev_asso,
                                                                        gt_ids=gt_ids,
                                                                        hypo_ids=hypo_ids, 
                                                                        tracks=self.active_targets,
                                                                        colsoftmaxed=softmaxed_col,
                                                                        toUpdate=True)


                    sum_dist, matched_objects = deepMOTPperFrame(distance_gt, motp_mask)
                    
                    total_objects = float(distance_gt.size(2))

                    if int(matched_objects) != 0:
                        motp = sum_dist/float(matched_objects)
                    else:
                        motp = torch.zeros(1).to(self.device)
                    total_objects = float(distance_gt.size(2))

                    if total_objects != 0:
                        #sshould put false negative ??????? if new detection for new track then will count as fn!
                        ##fn already in motp since distance should be small!!

                        mota = ( fn+ fp + 2*mm) / total_objects
                    else :
                        mota = torch.zeros(1).to(self.device)
                        
                    loss = 5*(motp ) + mota# + inv_motp/float(total_objects)
                else :
                    loss = torch.zeros(1).to(self.device)
                if loss.item() > 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.target_lstm.parameters(), 5.0)
                    self.optimizer.step()

                    to_ret = loss


            living = []
            for i in range(gts.shape[1]):
                for j in range(len(self.active_targets)):
                    if (gts[0][i,1] == self.active_targets[j].gt_id)  :
                        self.add_cues_to_target(target = self.active_targets[j], new_det = gts[0][i],
                                           dets = gts[0], frame = frame,new_bb = gts_apps[i],new_int = gts_ints[i])
                        living += [gts[0][i,1]]

            for i in range(gts[0].shape[0]):
                if gts[0][i,1] not in living:
                    self.add_target_train_not_start(gt=gts[0][i],frame=frame,gts=gts[0],
                                                    new_bb=gts_apps[i],new_int=gts_ints[i])

            self.prev_asso = self.update_asso(self.prev_asso)
            return to_ret
    
    
    def validation_step(self,gts,frame,epoch,i):
        with torch.no_grad():
            if i ==0:
                if gts.shape[1] > 0:
                    self.sort_to_birth_train(gts=gts[0], gts_to_birth=gts[0], frame=frame)
                    self.prev_asso = self.update_asso(self.prev_asso)
                    return 0
                
            else:
                to_ret = -1
                if gts.shape[1] > 0:

                    if(len(self.active_targets)):
                        # if we have targets track them
                        ##for all targets compute their distance to the gt detections

                        similarity_gt = []
                        gts_ints = []
                        gts_apps = []
                        
                        for gt in gts[0]:
                            gts_ints += [get_occupancy_grid(detection=gt, detection_list=gts[0],
                                             frame_width= frame.size[0],frame_height=frame.size[1],
                                             grid_height=self.grid_height,grid_width=self.grid_width,
                                             subgrid_height=self.subgrid_height,
                                            subgrid_width=self.subgrid_width).to(self.device)]

                            gts_apps += [self.cnn(get_bb_content_pil_opened(frame=frame,
                                                                    detection=gt).unsqueeze(0).to(self.device)).to(self.device)]
                        

                        for i,target in enumerate(self.active_targets):
                            similarity_gt_for_t = torch.zeros(( gts.shape[1],))
                            for idx in range(gts.shape[1]):
                                new_motion = get_velocity(det_t_minus_1=target.tracks[-1],
                                                          det_t=gts[0][idx]).to(self.device)
                                similarity_gt_for_t[idx] = self.target_lstm(target.apps, gts_apps[idx],
                                                                            target.motions, new_motion,
                                                                            target.interactions, gts_ints[idx]
                                                                            )
                            similarity_gt.append(similarity_gt_for_t)

                        ## we want distance and in the shape (detections, trackers):
                        similarity_gt = torch.stack(similarity_gt)
                     #   similarity_gt = torch.transpose(similarity_gt, 0, 1)
                        
                        distance_gt = 1 - similarity_gt
                        distance_gt = distance_gt.unsqueeze(0).to(self.device).contiguous()
                        output_dhn = self.dhn(distance_gt).to(self.device)

                        softmaxed_row = rowSoftMax(output_dhn, scale=100.0, threshold=0.5).contiguous()
                        softmaxed_col = colSoftMax(output_dhn, scale=100.0, threshold=0.5).contiguous()
                        
                        fn = missedObjectPerframe(softmaxed_col)
                        fp = falsePositivePerFrame(softmaxed_row)

                        hypo_ids = [t.id for t in self.active_targets]
                        gt_ids = [int(gt_id) for gt_id in gts[0,:,1]]
                        mm,  motp_mask, self.prev_asso,self.active_targets = missedMatchErrorV3_tracktor(prev_id=self.prev_asso,
                                                                            gt_ids=gt_ids,
                                                                            hypo_ids=hypo_ids, 
                                                                            tracks=self.active_targets,
                                                                            colsoftmaxed=softmaxed_col,
                                                                            toUpdate=True)


                        sum_dist, matched_objects = deepMOTPperFrame(distance_gt, motp_mask)
                        
                        total_objects = float(distance_gt.size(2))
                        if int(matched_objects) != 0:
                            motp = sum_dist/float(matched_objects)
                         #   inv_motp_mask = copy.deepcopy(motp_mask)
                         #   inv_motp_mask[motp_mask == 0 ] = 1
                         #   inv_motp_mask[motp_mask == 1 ] = 0
                         #   inv_motp = ((1- distance_gt)*inv_motp_mask).to(self.device)
                         #   inv_motp = torch.sum(inv_motp.view(1, -1), dim=1)/float(total_objects)
                        else:
                            motp = torch.zeros(1).to(self.device)
                         #   inv_motp_mask = copy.deepcopy(motp_mask)
                         #   inv_motp_mask[motp_mask == 0 ] = 1
                         #   inv_motp_mask[motp_mask == 1 ] = 0
                         #   inv_motp = ((1- distance_gt)*inv_motp_mask).to(self.device)
                         #   inv_motp = torch.sum(inv_motp.view(1, -1), dim=1)/float(total_objects)
                            
                        
                        if total_objects != 0:
                            #sshould put false negative ??????? if new detection for new track then will count as fn!
                            ##fn already in motp since distance should be small!!
                            mota = ( fn+  fp + 2*mm) / total_objects
                        else :
                            mota = torch.zeros(1).to(self.device)
                                
                        loss = 5*(motp ) + mota #+ inv_motp/float(total_objects*5)
                    else :
                        loss = torch.zeros(1).to(self.device)
                    if loss.item() > 0 :
                        to_ret = loss.item()
                # print results #
                    
                living = []
                for i in range(gts.shape[1]):
                    for j in range(len(self.active_targets)):
                        if (gts[0][i,1] == self.active_targets[j].gt_id)  :
                            self.add_cues_to_target(target = self.active_targets[j], new_det = gts[0][i],
                                               dets = gts[0], frame = frame,new_bb = gts_apps[i],new_int = gts_ints[i])
                            living += [gts[0][i,1]]

                for i in range(gts.shape[1]):
                    if gts[0][i,1] not in living:
                        self.add_target_train_not_start(gt=gts[0][i],frame=frame,gts=gts[0],
                                                        new_bb=gts_apps[i],new_int=gts_ints[i])
                self.prev_asso = self.update_asso(self.prev_asso)

                return to_ret


    def add_cues_to_target(self, target, new_det, dets, frame,new_bb,new_int):
        new_v = get_velocity(det_t_minus_1=target.tracks[-1], det_t=new_det)
        target.add_cues(new_bb,new_v,new_int)
        target.tracks += [new_det]
        
        
    def update_asso(self,asso):
        for t in self.active_targets:
            asso[t.gt_id] = t.id
        return asso
    
    
    def sort_to_die(self,ids_to_die):
        for gt in ids_to_die:
            if self.active_targets[gt].not_tracked_counter == self.threshold_to_kill:
                self.active_targets.pop(gt)
            else :
                self.active_targets[gt].not_tracked_counter += 1
    
    
    def sort_to_birth_train(self, gts, gts_to_birth,frame):
        for gt in gts_to_birth:
            self.add_target_train(gt,frame,gts)
        
        
    def add_target_train_not_start(self,gt,frame,gts,new_bb,new_int):
        motion = torch.zeros((2,)).to(self.device)
        gt_id = gt[1].item()
        new_track = Track(t_id=len(self.targets),
                          seq_length=self.seq_length,
                          device = self.device, gt_id = gt_id)
        new_track.add_cues(new_bb,motion,new_int)
        new_track.add_track(gt)
        self.targets += [new_track]
        self.active_targets += [new_track]

    def add_target_train(self,gt,frame,gts):
        with torch.no_grad():
            app=self.cnn(get_bb_content_pil_opened(frame=frame,
                                                   detection=gt).unsqueeze(0).to(self.device)).to(self.device)
        motion = torch.zeros((2,)).to(self.device)
        interaction = get_occupancy_grid(detection=gt, detection_list=gts,
                                             frame_width= frame.size[0],frame_height=frame.size[1],
                                             grid_height=self.grid_height,grid_width=self.grid_width,
                                             subgrid_height=self.subgrid_height,subgrid_width=self.subgrid_width)
        gt_id = gt[1].item()
        new_track = Track(t_id=len(self.targets),
                          seq_length=self.seq_length,
                          device = self.device, gt_id = gt_id)
        new_track.add_cues(app,motion,interaction)
        new_track.add_track(gt)
        self.targets += [new_track]
        self.active_targets += [new_track]

    def track_or_inactive(self, detection):
        pred = self.active_svm.predict(self.active_features)
        return pred >= 0
    
    
    def check_terminate_target(self,target):
        if target.not_tracked_counter == self.threshold_to_kill:
            self.active_targets.remove(target)
    
    
                
class Track:
    def __init__(self,t_id,seq_length,device,gt_id =None):
        self.id = int(t_id)
        self.tracks = []
        self.apps = []
        self.motions = []
        self.interactions = []
        self.not_tracked_counter = 0
        self.seq_length = seq_length
        self.device = device
        if gt_id is not None :
            self.gt_id = int(gt_id)
        else :
            self.gt_id = None
    def add_cues(self,app,motion,interaction):
        if len(self.apps) == 2*self.seq_length -1 :
            self.apps.pop(0)
            self.motions.pop(0)
            self.interactions.pop(0)
        
        self.apps.append(app.to(self.device)) 
        self.motions.append( motion.to(self.device))
        self.interactions.append(interaction.to(self.device))

    def add_track(self,track):
        self.tracks.append(track)
        

    
              
        
        
        
        
        
    
        
