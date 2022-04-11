from queue import LifoQueue
from collections import  defaultdict

import numpy as np
import copy
import sklearn
from scipy.optimize import linear_sum_assignment
from utils import overlap,get_velocity,get_bb_content_pil,get_bb_content_pil_opened,get_occupancy_grid,get_bb_center,nms,interpolate_tracks

import torch
from spatial_attention import *
from data_utils import dataset_utils
from lk_tracker import *
from lk_utils import *
from libsvm.python.svm import *
from libsvm.python.svmutil import *

from spatial_temporal_attention_network import *
class Tracker:
    def __init__(self, target_lstm=None, cnn =None, active_svm=None, batch_size =1, seq_length=6, grid_height=15, grid_width=15, subgrid_height=7, subgrid_width=7, threshold_association=0.8, max_height=1,max_width=1,max_score=1,
                 threshold_overlap=0.3,threshold_to_kill=50,threshold_reid=0.3,threshold_confidence =0.5,
                 threshold_conflict=0.6, minimum_confidence =10,device = 'cuda'):
        
        self.targets = []
        self.active_targets = []
        self.target_lstm = target_lstm
        self.device = device
        self.cnn = cnn
        self.active_svm = active_svm
        self.seq_length = seq_length
        self.prev_asso = {} 
        self.batch_size = batch_size
        self.threshold_association = threshold_association
        self.threshold_overlap = threshold_overlap
        self.threshold_to_kill = threshold_to_kill
        self.threshold_reid = threshold_reid
        self.threshold_confidence = threshold_confidence
        self.threshold_conflict = threshold_conflict
        self.minimum_confidence = minimum_confidence
        self.grid_height= grid_height
        self.grid_width = grid_width
        self.subgrid_height = subgrid_height
        self.subgrid_width = subgrid_width
        self.max_width =max_width
        self.max_height =max_height
        self.max_score = max_score
        self.new_track_id =1
        
    '''
    def track(self, video_seq, detections):
    '''
    def reset(self):
        for t in self.targets:
            del t

        self.targets = []
        self.active_targets = []
        self.new_track_id =1

    def testing_step(self,dets,frame,frame_source):
        with torch.no_grad():
            self.frame_source = frame_source
                
            dets = dets[dets[:,6]>self.minimum_confidence]

            dets_ints = []
            input_to_cnn = []
            dets_apps = []

            if dets.shape[0] > 0:
                dets = torch.Tensor(nms(np.array(dets)))



            for det in dets:
                dets_list_for_int = ([torch.Tensor(t_past.tracks[-1]) for t_past in self.active_targets])
                if not len(dets_list_for_int):
                    dets_list_for_int =dets
                else :
                    dets_list_for_int = torch.stack(dets_list_for_int)
                dets_ints += [get_occupancy_grid(detection=det, detection_list=dets_list_for_int,
                                 frame_width= frame.size[0],frame_height=frame.size[1],
                                 grid_height=self.grid_height,grid_width=self.grid_width,
                                 subgrid_height=self.subgrid_height,
                                subgrid_width=self.subgrid_width).to(self.device)]
                if self.cnn is not None:
                    dets_apps += [self.cnn(get_bb_content_pil_opened(frame=frame, detection=det).to(self.device).unsqueeze(0))]
                else :
                    dets_apps += [torch.zeros((1,1))]

            long_tracks = [ long_t for long_t in self.active_targets if len(long_t.tracks) > 10]
            short_tracks = [short_t for short_t in self.active_targets if len(short_t.tracks) <=10]

            long_tracks_sorted = sorted(long_tracks, key=lambda x: (len(x.tracks),len(x.state)), reverse=True)
            short_tracks_sorted = sorted(short_tracks, key=lambda  x: (len(x.tracks),len(x.state)), reverse=True)


            all_tracks = [long_tracks_sorted,short_tracks_sorted]

            all_tracks_cat = sorted(self.active_targets, key=lambda x: (len(x.tracks),len(x.state)), reverse=True)
           # all_tracks_cat = self.active_targets
            for tracks_idx, tracks in enumerate(all_tracks):
                for target_idx, target in enumerate(tracks):
                    if target.state == 'tracked':
                        target.track(frame=frame,detections=dets,frame_source = self.frame_source)
                        
                        if dets.shape[0] >0:
                            if target.state == 'lost':
                                if tracks_idx == 0:
                                    active_t = [t for t in  tracks[:target_idx] if t.state =='tracked']

                                if tracks_idx == 1:
                                    active_t = [t for t in all_tracks[0] if t.state =='tracked']
                                    active_t.extend([t for t in tracks[:target_idx] if t.state =='tracked'])

                                temp_dets,temp_apps,temp_ints = self.get_untracked_dets(active_t,
                                                                                        dets,dets_apps,dets_ints)

                                if len(temp_dets):
                                    target.associate(detections=temp_dets,frame=frame,
                                                     dets_apps=temp_apps,dets_ints=temp_ints,
                                                     frame_source = self.frame_source)
                        

                active_t = [t for t in all_tracks[0] if t.state =='tracked']
                if tracks_idx ==1:
                    active_t.extend( [t for t in all_tracks[1] if t.state =='tracked'])

                temp_dets,temp_apps,temp_ints = self.get_untracked_dets(active_t,dets,dets_apps,dets_ints)
                if tracks_idx == 0 :
                    lost_t = [t for t in all_tracks[0] if t.state =='lost']
                if tracks_idx ==1 :
                    lost_t = [t for t in all_tracks[1] if t.state =='lost']
                if len(temp_dets):
                    self.hungarian_assignment(lost_t,temp_dets,temp_ints,temp_apps,frame)

            active_t = [t for t in self.active_targets if t.state == 'tracked']
            unsuppressed_dets = self.suppress_tracked(dets,active_t)
            ### for every target not assigned increment inactive counter and mayb terminate
            for i in reversed(range(len(self.active_targets))):
                if self.active_targets[i].state != 'tracked':
                    self.active_targets[i].not_tracked_counter += 1
                    if self.active_targets[i].not_tracked_counter == self.threshold_to_kill:
                        self.active_targets.pop(i)

            ### for every detection not assigned see if should start a target
            for i,detection in enumerate(dets):
                if any([(detection == d).all() for d  in  unsuppressed_dets]) :
                    if self.track_or_inactive(detection,frame):
                        self.add_target(detection = detection, frame = frame,
                                dets = dets,new_app = dets_apps[i], new_int = dets_ints[i])


            if self.threshold_conflict >0:
                self.resolve_conflicts(dets)

            self.delete_gone_targets(frame)

            return 0
                    
    
    def hungarian_assignment(self, targets, detections, dets_ints, dets_apps, frame):
       
        similarity = []
        
        if detections.shape[0] == 0:
            return
        for target in targets:
            similarity.append(target.get_similarity_vector(detections,dets_apps,dets_ints ))
        
        if not len(similarity):
            return
        similarity = torch.stack(similarity)
        
        distance = 1 - similarity
        
        row_ind, col_ind = linear_sum_assignment(distance)

        for r, c in zip(row_ind, col_ind):
            if distance[r, c] <= self.threshold_reid:
                target = targets[r]
                self.add_cues_to_target(target = target,
                                        new_det = detections[c], dets = detections, frame = frame,
                                        new_app = dets_apps[c],new_int = dets_ints[c])
                target.lk_tracker.update(frame,detections[c],self.frame_source)
        return
        
        
        
    def get_untracked_dets(self,active_targets,detections,dets_apps,dets_ints):
        
        temp_dets = self.suppress_tracked(detections,active_targets)
        if temp_dets.nelement() == 0:
            return [],[],[]

        temp_dets_idxs  = []
        for i,temp_det in enumerate(temp_dets):
            for j,det in enumerate(detections):
                if (temp_det == det).all():
                    temp_dets_idxs.append(j)

        temp_apps = [t_d_app for i,t_d_app in enumerate(dets_apps) if i in temp_dets_idxs ]
        temp_ints = [t_d_int for i,t_d_int in enumerate(dets_ints) if i in temp_dets_idxs ]

        return temp_dets,temp_apps,temp_ints
        
        
        
    def add_cues_to_target(self, target, new_det, dets, frame,new_app,new_int):
        new_v = get_velocity(det_t_minus_1=target.tracks[-1], det_t=new_det)
        target.add_cues(new_app,new_v,new_int)
        target.add_track(new_det)
        target.old_not_tracked_counter = target.not_tracked_counter
        target.not_tracked_counter = 0

        
    def normalize_active_features(self,det, frame_width, frame_height, max_width, max_height, max_score):
        norm_det = np.zeros_like(det)
        norm_det[2] = det[2] / frame_width
        norm_det[3] = det[3] / frame_height
        norm_det[4] = det[4] / max_width
        norm_det[5] = det[5] / max_height
        norm_det[6] = det[6] / max_score
        return norm_det[2:7]

    def sort_to_die(self,ids_to_die):
        for gt in ids_to_die:
            if self.active_targets[gt].not_tracked_counter == self.threshold_to_kill:
                self.active_targets.pop(gt)
            else :
                self.active_targets[gt].not_tracked_counter += 1
    
    
    def add_target(self,detection,frame,dets,new_app,new_int):
        motion = torch.zeros((2,))
        new_track = Track(t_id=self.new_track_id,
                          seq_length=self.seq_length,frame_source = self.frame_source,
                          device = self.device,frame=frame,detection=detection,cnn = self.cnn,target_lstm = self.target_lstm)
        new_track.add_cues(new_app,motion,new_int)
        self.targets += [new_track]
        self.active_targets += [new_track]
        self.new_track_id +=1
        
        
    def add_target_start(self,detection,frame,dets):
        if self.cnn is not None:
           # app = self.cnn(get_bb_content_pil_opened(frame=frame,detection=detection).unsqueeze(0).to(self.device))
            app = self.cnn(get_bb_content_pil_opened(frame=frame, detection=detection).to(self.device).unsqueeze(0))
        else :
            app = torch.zeros((1,1))
        motion = torch.zeros((2,))
        interaction = get_occupancy_grid(detection=detection,
                                         detection_list=dets,
                                         frame_width= frame.size[0],frame_height=frame.size[1],
                                         grid_height=self.grid_height,grid_width=self.grid_width,
                                             subgrid_height=self.subgrid_height,subgrid_width=self.subgrid_width)
        new_track = Track(t_id=self.new_track_id,
                          seq_length=self.seq_length,frame_source =self.frame_source,
                          device = self.device,frame=frame,
                          detection=detection,cnn = self.cnn,target_lstm = self.target_lstm)
        
        new_track.add_cues(app,motion,interaction)
        self.targets += [new_track]
        self.active_targets += [new_track]
        self.new_track_id +=1

        
    def track_or_inactive(self, detection,frame):
        new_features = self.normalize_active_features(det=detection,frame_width = frame.size[0],
                                                      frame_height = frame.size[1],
                                                      max_width=self.max_width,
                                                      max_height=self.max_height, 
                                                      max_score=self.max_score)
        if type(self.active_svm) == sklearn.svm._classes.SVC:
            pred = self.active_svm.predict([[*new_features,1]])
            
        elif str(type(self.active_svm)) == "<class 'svm.svm_model'>":
             pred =  svm_predict([1],[[*new_features,1]],self.active_svm,'-q')[0]
                
        return pred[0]>0
    
    
    
    def suppress_asso(self,detections):
        ### removes assos take shouldnt exist because the height is >2*height_previous
        impossible_asso = []
        
        for t_idx, t in enumerate(self.active_targets):
            h_t = t.tracks[-1][5]
            for d_idx, d in enumerate(detections) :
                h_d = d[5]
                h_ratio =  h_t/h_d
                min_ratio = min(h_ratio,1/h_ratio)
                if min_ratio < self.threshold_association:
                    impossible_asso.append((t_idx,d_idx))
        return impossible_asso
    
    
    
    def suppress_tracked(self,detections,active_t):
        det_tracked = []
        for idx,t_tracked in enumerate(active_t):
            det_tracked.append(np.array(t_tracked.tracks[-1]))
        
        if not det_tracked :
            return detections
        
        det_tracked = np.stack(det_tracked)
        
        not_tracked_dets = []
        for idx,det in enumerate(detections):
            iou,iod,iog = overlap(np.array(det),det_tracked)
            max_iou = np.max(iou)
            sum_iod = np.sum(iod)
            
            if  not (max_iou > 0.5 or sum_iod > 0.5):
                not_tracked_dets.append(det)
                
        if len(not_tracked_dets):
            return  torch.stack(not_tracked_dets)
        else :
            return torch.Tensor([])
                                   
        


    def resolve_conflicts(self,detections):
        suppress_targets = []
        
        t_tracks = []
        t_tracks_idxs = []
        idx_in_map = 0
        for i,t in enumerate(self.active_targets):
            if t.state == 'tracked':
                t_tracks_idxs.append(i)
                t_tracks.append(np.array(t.tracks[-1]))
        if not t_tracks:
            return
        t_tracks = np.stack(t_tracks)
        removed_ts = set()
        for t_idx,track in enumerate(t_tracks) :
            if t_idx in removed_ts:
                continue
            _,iod,_ = overlap(track,t_tracks)
            
            iod[t_idx] = 0
            max_iod = np.max(iod)
            max_iod_idx = np.argmax(iod)
            
            if max_iod > self.threshold_conflict:
                streak_i = self.active_targets[t_tracks_idxs[t_idx]].not_tracked_counter
                streak_j =  self.active_targets[t_tracks_idxs[max_iod_idx]].not_tracked_counter
                
                ov1,_,_ = overlap(track,detections)
                ov2,_,_ = overlap(t_tracks[max_iod_idx],detections)
                ov1 = np.max(ov1)
                ov2 = np.max(ov2)
                
                if streak_i > streak_j:
                    to_supr = max_iod_idx
                elif streak_i < streak_j:
                    to_supr = t_idx
                else:
                    if ov1 > ov2:
                        to_supr = max_iod_idx
                    else:
                        to_supr = t_idx
                
                self.active_targets[t_tracks_idxs[to_supr]].state = 'lost'
                
                removed_ts.add(to_supr)
        to_delete_tracks = set()
        for rem_ts_idx in removed_ts:
            self.active_targets[t_tracks_idxs[rem_ts_idx]].remove_one_timestep()
            if len(self.active_targets[t_tracks_idxs[to_supr]].tracks) ==0:
                to_delete_tracks.add(t_tracks_idxs[to_supr])
                
        for del_idx in sorted(to_delete_tracks,reverse=True):
            self.active_targets.pop(del_idx)
            
    def delete_gone_targets(self,frame):
        to_delete = list()
        for t_idx,target in enumerate(self.active_targets):
            
            _,iod,_ = overlap(np.array(target.tracks[-1]),
                              np.expand_dims(np.array([-1,-1,1,1,*frame.size]),axis=0))
            
            if iod < 0.3:
                to_delete.append(t_idx)
        for t_idx in sorted(to_delete,reverse=True):
            self.active_targets.pop(t_idx)
                
class Track:
    def __init__(self,t_id,seq_length,device,frame,detection,frame_source, gt_id =None,cnn = None,target_lstm = None):
        self.id = int(t_id)
        self.tracks = []
        self.apps = []
        self.motions = []
        self.interactions = []
        self.not_tracked_counter = 0
        self.old_not_tracked_counter=0
        self.seq_length = seq_length
        self.device = device
        self.state = 'lost'
        self.lk_tracker = LK_tracker(frame=frame,
                                     detection = np.array(detection),frame_source=frame_source)
        self.add_track(detection)
        self.grid_height = 15
        self.subgrid_height = 7
        self.cnn = cnn
        self.target_lstm = target_lstm
        
        if gt_id is not None :
            self.gt_id = int(gt_id)
        else :
            self.gt_id = None
            
    def add_cues(self,bb,motion,interaction):
        
        if len(self.apps) == 2*self.seq_length - 1:
            ## all have same length
            self.motions.pop(0)
            self.interactions.pop(0)
        
        self.old_not_tracked_counter= self.not_tracked_counter
        self.not_tracked_counter = 0
        self.apps.append( bb.to(self.device))
        self.motions.append( motion.to(self.device))
        self.interactions.append(interaction.to(self.device))
        self.lk_tracker.tracks = copy.deepcopy(self.tracks)
        
    def add_track(self,track):
        track[1] = self.id
        self.tracks.append(torch.Tensor(np.asarray(track,dtype=float)))
        self.lk_tracker.tracks = copy.deepcopy(self.tracks)
        self.state = 'tracked'

    def remove_one_timestep(self):
        if len(self.tracks):
            self.tracks.pop(-1)
            self.apps.pop(-1)
            self.motions.pop(-1)
            self.interactions.pop(-1)
            
        self.old_not_tracked_counter +=1
        self.not_tracked_counter = self.old_not_tracked_counter
    
    
    def track(self,frame,detections,frame_source):
        new_bb,flag = self.lk_tracker.track(frame,np.array(detections),frame_source)
        if  flag != 1:
            self.state = 'lost'
            
        else:
            if detections.shape[0] ==0:
                frame_num = self.tracks[-1][0] +1 
            else :
                frame_num = detections[0][0]
            temp_det = np.array([frame_num,-1000,*new_bb,50,-1,-1,-1])
            
            new_motion = get_velocity(det_t_minus_1=self.tracks[-1], 
                                                              det_t=temp_det).to(self.device)
                       
            new_int = get_occupancy_grid(detection=temp_det, detection_list=detections,
                                             frame_width= frame.size[0],frame_height=frame.size[1],
                                             grid_height=self.grid_height,grid_width=self.grid_height,
                                             subgrid_height=self.subgrid_height,
                                             subgrid_width=self.subgrid_height)
            
            new_app = get_bb_content_pil_opened(frame=frame, detection=temp_det).to(self.device).unsqueeze(0)
            
            if self.cnn != None:
                new_app = self.cnn(new_app)
            self.add_cues(new_app,new_motion,new_int)
            self.add_track(temp_det)
            
    
    def suppress_height_dist(self,detections):
        if detections.shape[0] == 0:
            return []
        
        
        impossible_asso = []
        h_t = self.tracks[-1][5]
        
        ctarget = motion_prediction(self,detections[0][0])
        for d_idx, d in enumerate(detections) :
            h_d = d[5]
            h_ratio =  h_t/h_d
            min_ratio = min(h_ratio,1/h_ratio)
            
            cdet = np.array([d[2]+d[4]/2,d[3]+d[5]/2])
            distance  = np.linalg.norm( cdet - ctarget,ord=2)/self.tracks[-1][4]
            if min_ratio < 0.6 or distance > 3:
                impossible_asso.append(d_idx)
                       
        
        
        return impossible_asso
    
    
    def get_similarity(self, app, inter,motion):
        if self.cnn is not None:
            similarity = self.target_lstm(self.apps, 
                                                app,
                                                self.motions,
                                                motion,
                                                self.interactions,
                                                inter
                                                )
        else:
            similarity = self.target_lstm(None,
                                          None,
                                          self.motions,
                                          motion,
                                          self.interactions,
                                          inter)
        return similarity
    
    def get_similarity_vector(self,detections,dets_apps,dets_ints):
        similarity= torch.zeros((detections.shape[0],))

        impossible_asso = self.suppress_height_dist(detections)
        for det_idx,det in enumerate(detections):
            #0 similarity to asso that have unrelated heights 
            if det_idx in impossible_asso :
                similarity[det_idx]=0
            else :
                new_motion = get_velocity(det_t_minus_1=self.tracks[-1], 
                                          det_t=detections[det_idx]).to(self.device)
                similarity[det_idx]= self.get_similarity(dets_apps[det_idx],dets_ints[det_idx],new_motion)

        return similarity
        
    def associate(self,detections,frame,dets_apps,dets_ints,frame_source):
        similarity = self.get_similarity_vector(detections,dets_apps,dets_ints)
        max_sim_idx = torch.argmax(similarity)
        max_sim = torch.max(similarity)
        
        if max_sim > 0.5:
            new_motion = get_velocity(det_t_minus_1=self.tracks[-1], 
                          det_t=detections[max_sim_idx]).to(self.device)
            
            app = dets_apps[max_sim_idx]
            
            interpolate_tracks(self,detections[max_sim_idx])
            self.add_cues(app,new_motion,dets_ints[max_sim_idx])
            self.add_track(detections[max_sim_idx])
            self.lk_tracker.update(frame,detections[max_sim_idx],frame_source)
    
        
