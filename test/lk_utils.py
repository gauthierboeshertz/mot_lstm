import numpy as np
import copy
from utils import *
import matplotlib.pyplot as plt
import cv2

import scipy 

def bb_rescale_relative(bbox,s):
    s1 = 0.5*(s[0]-1)*(bb_width(bbox))
    s2 = 0.5*(s[1]-1)*(bb_height(bbox))
    new_det = copy.deepcopy(bbox)
    new_det[0] += -s1
    new_det[1] += -s2
    new_det[2] += s1
    new_det[3] += s2
    return new_det


def bb_height(bb):
    return bb[3]-bb[1] +1

def bb_width(bb):
    return bb[2]-bb[0] +1

def bb_shift_absolute(bbox, s):
    
    new_bbox = copy.deepcopy(bbox)
    return np.array([new_bbox[0]+s[0],new_bbox[1]+s[1],new_bbox[2]+s[0],new_bbox[3]+s[1]])


def motion_prediction(tracker,frame_num):
    
    last_n = min(10,len(tracker.tracks))
    past_c = [get_bb_center(det) for det in (tracker.tracks[-last_n:])]
    past_fr = [det[0] for det in (tracker.tracks[-last_n:])]
    
    vx = 0
    vy = 0
    
    count =0
    for i in range(1,last_n):
        vx += (past_c[i][0] - past_c[i-1][0])/(past_fr[i] - past_fr[i-1])
        vy += (past_c[i][1] - past_c[i-1][1])/(past_fr[i] - past_fr[i-1])
        count+=1
    
    if count>0:
        vx/=count
        vy/=count
    
    if len(past_fr) <= 1:
        ret_cx,ret_cy = get_bb_center(tracker.tracks[-1])
    else:
        last_cx,last_cy = get_bb_center(tracker.tracks[-1])
        ret_cx  = last_cx + vx *( frame_num +1 - tracker.tracks[-1][0])
        ret_cy  = last_cy + vy *( frame_num +1 - tracker.tracks[-1][0])
        
       
    return ret_cx,ret_cy



def min_crop(frame,bb):
    
    w = int(bb_width(bb))
    h = int(bb_height(bb))
    
    crop = np.zeros((h,w),dtype= 'uint8')
    x1 = max(0,bb[0])
    y1 = max(0,bb[1])
    x2 = min(frame.shape[1],bb[2])
    y2 = min(frame.shape[0],bb[3])
    
    patch = frame[int(y1):int(y2),int(x1):int(x2)]
    x1 = x1 - bb[0] 
    y1 = y1 - bb[1] 
    x2 = x2 - bb[0] 
    y2 = y2 - bb[1] 
    
    crop[int(y1):int(y2),int(x1):int(x2)] = patch
    return crop
    
 

def lk_crop_bbox(BB,frame_source,std_box=[30,60],enlarge_box = np.array([5,3])):
    
    x = BB[2]
    y = BB[3]
    x2 = BB[4]
    y2 = BB[5]

    s = np.array([std_box[0]/(bb_width(BB[2:6])), std_box[1]/(bb_height(BB[2:6]))])
    
    x_scaled = x*s[0]
    y_scaled = y*s[1]
    x2_scaled = x_scaled + std_box[0] - 1
    y2_scaled = y_scaled + std_box[1] - 1    
    
    bb_scale = np.ceil(np.array([x_scaled,y_scaled,x2_scaled,y2_scaled]))
    
    frame = cv2.imread(frame_source,0)
    new_frame = cv2.resize(frame, (int(np.round(frame.shape[1]*s[0])),int(np.round(frame.shape[0]*s[1]))))
    
    bb_ret = (bb_rescale_relative(bb_scale,enlarge_box))
    new_crop = min_crop(new_frame,bb_ret)

    ret_bb_crop = bb_shift_absolute(bb_scale,np.array([-bb_ret[0],-bb_ret[1]]))
                                    
    return new_crop, ret_bb_crop, bb_ret, s

def bb_points(bb, num_x_points,num_y_points,margin=[5,2]):
    bb[0]+= margin[0]
    bb[1] += margin[1]
    bb[2] -= margin[0]
    bb[3] -= margin[1]
    
    if num_y_points ==1 and num_x_points==1:
        return get_bb_center(np.array([-1,-1,*bb]))
    
    if num_y_points == 1 and num_x_points>1:
        return np.hstack((  np.expand_dims(np.linspace(bb[0],bb[0]+bb[2],num_x_points),1),np.ones((num_x_points,1))*bb[1]))
                                    
    if num_x_points == 1 and num_y_points>1:
        return np.hstack((np.ones((num_y_points,1))*bb[0],np.expand_dims(np.linspace(bb[1],bb[1]+bb[3],num_y_points),1)))
    
    else:
        step_x = (bb[2]-bb[0])/(num_x_points-1)
        step_y = (bb[3] - bb[1])/(num_y_points-1)
        x_pts = np.linspace(bb[0],bb[2],num_x_points)
        y_pts = np.linspace(bb[1],bb[3],num_y_points)
        return np.transpose([np.tile(x_pts, len(y_pts)), np.repeat(y_pts, len(x_pts))])
    
def bb_predict(bbox,pt0,pt1):
    
    of = pt1 - pt0
    dx = np.median(of[:,0])
    dy = np.median(of[:,1])
    
    d0 = scipy.spatial.distance.pdist(pt0)
    d1 = scipy.spatial.distance.pdist(pt1)
    
    s = np.median(d1/d0)
    
    s1 = 0.5*(s-1)*(bb_width(bbox))
    s2 = 0.5*(s-1)*(bb_height(bbox))
    
    new_bb = np.array([bbox[0]-s1 + dx, bbox[1]- s2 + dy, bbox[2]+s1+dx,bbox[3]+s2+dy])
    return new_bb , np.array([s1,s2])





            
             
def bb_isout(bb,img):
    return bb[0] > img.shape[1] or bb[1] > img.shape[0]
    
def normcrosscorrelation(frame_i,frame_j,status,pts_i,pts_j):
    match = np.zeros((pts_i.shape[0],))
    for i in range(pts_i.shape[0]) :
        if (status[i] == 1) :
            pixi = cv2.getRectSubPix( frame_i,(10,10), (pts_i[i][0][0],pts_i[i][0][1]) )
            pixj = cv2.getRectSubPix( frame_j, (10,10), (pts_j[i][0][0],pts_j[i][0][1]) )
            match[i] = cv2.matchTemplate( pixi,pixj,  cv2.TM_CCOEFF_NORMED )[0][0]
        else :
            match[i] = 0.0
    return match
        
    
def calc_lk(img_i,img_j,bb_i,bb_j,margin=[5,2],level=2,threshold_error=0.5):
    

    lk_params_fp = dict( winSize  = (4,4),
                        maxLevel =1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              20, 0.03),
                        flags=4
                       )
    
    lk_params_bp = dict( winSize  = (4,4),
                        maxLevel =1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              20, 0.03),
                        flags =4
                  )
    
    
    i_points = np.expand_dims(bb_points(copy.deepcopy(bb_i),10,10,margin),axis=1)
    i_points = np.asarray(i_points,dtype=np.float32)
    
    j_points = np.expand_dims(bb_points(copy.deepcopy(bb_j),10,10,margin),axis=1)
    j_points = np.asarray(j_points,dtype=np.float32)
    
    
    bp = copy.deepcopy(i_points)
    fp, stp, err = cv2.calcOpticalFlowPyrLK(img_i, img_j, copy.deepcopy(i_points), cv2.UMat(copy.deepcopy(j_points)),  **lk_params_fp)

    bp, stb, err = cv2.calcOpticalFlowPyrLK(img_j, img_i, copy.deepcopy(fp.get()), cv2.UMat(copy.deepcopy(i_points)), **lk_params_bp)

    fp = fp.get()
    bp = bp.get()
    return i_points,fp,stp,bp,stb
    

def lk(img_i,img_j,bb_i,bb_j,margin=[5,2],level=2,threshold_error=0.5):
    
    if img_i.shape != img_j.shape:
        return None,None,None,3
    i_points,fp,stp,bp,stb = calc_lk(img_i,img_j,bb_i,
                            bb_j,margin,level,threshold_error)
    

    bb3 = []
    stb = stb.get()
    bp_st =   np.expand_dims(bp[(stb==1 )],axis=1)
    fp_st = np.expand_dims(fp[(stb==1 )],axis=1)
    i_points_st = np.expand_dims(i_points[(stb==1 ) ],axis=1)
    
    error = np.sqrt(np.power(bp - i_points, 2).sum(axis=2))
    ncc  = normcrosscorrelation(img_i,img_j,stb,i_points,fp)
    
    ncc = np.expand_dims(ncc,axis=1)
    
    error[stb != 1] = np.nan
    ncc[stb != 1] = np.nan
    
    
    medFB = np.nanmedian(error)
    medNCC = np.nanmedian(ncc)
    
    if stb[stb==1].shape[0] < 4:
        flag = 3
        return None,None,None,flag    
        
   # temp_med = np.median(temp_error)
    reliable_points = (error <= medFB) & (ncc >= medNCC)
    
    reliable_ipoints = np.expand_dims(i_points[reliable_points],axis=1)
    
    reliable_fpoints = np.expand_dims(fp[reliable_points],axis=1)
    
    if reliable_fpoints.shape[0] < 4:
        flag = 3
        return None,None,None,flag
    
    
    
    index_left = reliable_ipoints[:,:,0] < ( bb_i[0]+bb_i[2])/2
    bb_left = reliable_fpoints[index_left]
    
    index_right = reliable_ipoints[:,:,0] >= ( bb_i[0]+bb_i[2])/2
    bb_right = reliable_fpoints[index_right]
    
    index_top= reliable_ipoints[:,:,1] < ( bb_i[1]+bb_i[3])/2
    bb_top = reliable_fpoints[index_top]
    
    index_bot = reliable_ipoints[:,:,1] >= ( bb_i[1]+bb_i[3])/2
    bb_bot= reliable_fpoints[index_bot]

    
    
    new_bb,_ = bb_predict(bb_i, np.squeeze(reliable_ipoints,axis=1),np.squeeze(reliable_fpoints,axis=1))
   # new_bb,_ = bb_predict(bb_i,i_points_st[(error <= medFB  )],reliable_points)
    
    if  bb_bot.size ==0 or bb_top.size ==0 or bb_left.size == 0 or bb_right.size ==0:
        flag = 3
        return new_bb,error,reliable_points,flag
    
    flag = 1;

    if medFB > 10:#1 or bb_isout(new_bb,img_j):
        flag = 3
        return new_bb,error,reliable_points,flag
    

    return new_bb,error,reliable_fpoints,flag

