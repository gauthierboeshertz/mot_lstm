import numpy as np
from PIL import Image

from lk_utils import *


class LK_tracker:
    
    def __init__(self,detection,frame, frame_source, box_enlarge=np.array([5,3])):
        
        
        self.box_enlarge = box_enlarge
        self.std_box = np.array([30, 60])

        self.overlaps =[1]
        self.update(frame,detection,frame_source)
        self.tracks = []
        
    def update(self,frame,detection,frame_source):
        self.template_det = detection
        self.template_bb = detection[2:6]
        det_for_crop = copy.deepcopy(detection)
        det_for_crop[4] = det_for_crop[4] + det_for_crop[2]
        det_for_crop[5] = det_for_crop[5] + det_for_crop[3]
        
        self.template_crop,self.template_bb, _, _ = lk_crop_bbox(BB = det_for_crop,
                                                        frame_source=frame_source)
        
        
    def track(self,frame,detections,frame_source):
        
        if detections.shape[0] == 0:
            frame_num = self.tracks[-1][0] + 1
        else:
            frame_num = detections[0][0]
        c_pred = motion_prediction(self,frame_num=frame_num)
        
        w = self.template_det[4]
        h = self.template_det[5]
        
        bb_pred_motion = np.array([(c_pred[0]-w/2),(c_pred[1]-h/2),(c_pred[0]+w/2),(c_pred[1]+h/2)])
        
        new_crop,new_bb, bb, s = lk_crop_bbox(BB = copy.deepcopy(np.array([-1,-1,*bb_pred_motion])),
                                             frame_source = frame_source)
        
        
        pred_bb,_,_,flag= lk(copy.deepcopy(self.template_crop),
                             copy.deepcopy(new_crop),
                             copy.deepcopy(self.template_bb),
                             copy.deepcopy(new_bb),
                             margin=[5,2],level=2,threshold_error=0.5)
        
        if flag != 1:
            return None,0

        mean_ov = np.mean(self.overlaps[-10:])
        
        if mean_ov < 0.8 :
            return None,0
        

        pred_bb = bb_shift_absolute(copy.deepcopy(pred_bb), np.array([bb[0],bb[1]]))
        pred_bb = np.array([pred_bb[0]/s[0],pred_bb[1]/s[1],pred_bb[2]/s[0],pred_bb[3]/s[1]])
        
        pred_bb[2] = pred_bb[2] - pred_bb[0]
        pred_bb[3] = pred_bb[3] - pred_bb[1]

        max_ov = 0

        if not detections.shape[0] ==0:
            ov = np.array(overlap(copy.deepcopy(np.array([-1,-1,*pred_bb])),copy.deepcopy(detections)))
            max_ov = np.max(ov)
            max_ov_idx = np.unravel_index(np.argmax(ov), ov.shape)
            

        if max_ov >0.5:
            max_det = detections[max_ov_idx[1]]
            max_det_bb = max_det[2:6]
            ret_bb = np.array([max_det_bb,pred_bb])
            ret_bb = np.mean(ret_bb,axis=0)
            
        else :
            ret_bb = pred_bb
        
        
        if max_ov> 0.5:
            self.overlaps.append(1)
        else:
            self.overlaps.append(0)
        return ret_bb,1
        
        
        
        
        
        
        
        
        
        
        