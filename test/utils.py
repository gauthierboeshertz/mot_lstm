import numpy as np
import csv
import cv2

import math
from PIL import Image
from torchvision.transforms import PILToTensor,ToTensor,Normalize
from torchvision import transforms
import torch
import torch
import motmetrics

from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)
def overlap(det, gt):
    iou = np.zeros((gt.shape[0]))
    iod = np.zeros((gt.shape[0]))
    iog = np.zeros((gt.shape[0]))
    
    det_x1 = det[2]
    det_y1 = det[3]
    det_x2 = det_x1 + det[4] - 1
    det_y2 = det_y1 + det[5] - 1

    gt_x1 = gt[:, 2]
    gt_y1 = gt[:, 3]
    gt_x2 = gt_x1 + gt[:, 4] - 1
    gt_y2 = gt_y1 + gt[:, 5] - 1

    det_area = det[4] * det[5]
    gt_areas = np.multiply(gt[:, 4], gt[:, 5])

    left = np.maximum(np.ones_like(gt_x1) * det_x1, gt_x1)  
    top = np.maximum(np.ones_like(gt_y1) * det_y1, gt_y1)
    right = np.minimum(np.ones_like(gt_x2) * det_x2, gt_x2)
    bot = np.minimum(np.ones_like(gt_y2) * det_y2, gt_y2)

    w = right - left + 1
    h = bot - top + 1

    bigger_than0 = np.where(np.multiply((w > 0) * w, (h > 0) * h) > 0)  ##take only boxes that intersect

    intersection = w[bigger_than0] * h[bigger_than0]
    union = det_area + gt_areas[bigger_than0] - intersection
    iou[bigger_than0] = intersection / union
    iod[bigger_than0] = intersection / det_area
    iog[bigger_than0] = intersection / gt_areas[bigger_than0]

    return iou, iod, iog



def nms(detections):
    
    x = detections[:,2] 
    y = detections[:,3] 
    x2 = detections[:,2] + detections[:,4]
    y2 = detections[:,3] + detections[:,5]

    areas = (x2 - x +1)*(y2-y+1)
    scores = detections[:,6]
    
    sorted_idx = np.argsort(scores) #highest score to lowest
    sorted_idx = np.flip(sorted_idx)
    ret_dets = np.ones((detections.shape[0],))
    
    for idx in sorted_idx:
        
        for higher_idx in sorted_idx[:idx]:
            if ret_dets[higher_idx] == 0:
                continue
           # idx = int(idx)
          #  higher_idx = int(higher_idx)
            """
            xi = np.max([x[idx],x[higher_idx]])
            yi = np.max([y[idx],y[higher_idx]])
            x2i = np.max([x2[idx],x2[higher_idx]])
            y2i = np.max([y2[idx],y2[higher_idx]])

            w = x2i -xi
            h = y2i - yi
            
            if w>0 and h>0:
                o = w*h/(areas[idx] +areas[higher_idx] - w*h)
                iod = w*h/(areas[idx] )
                iog = w*h/(areas[higher_idx] )
                
                if o > 0.6 or iod >0.95 or iog > 0.95:
                    ret_dets[idx]=0
                    break
            """
            iou,iod,iog = overlap(detections[idx], np.array([detections[higher_idx]]))
            if iou[0] > 0.6 or iod[0] >0.95 or iog[0] > 0.95:
                print('nmsed ')
                ret_dets[idx]=0
                break
    return detections[ret_dets ==1]


def interpolate_tracks(tracker,detection):
    if len(tracker.tracks):
        last_fr = tracker.tracks[-1][0]
        new_fr = detection[0]
        
        if new_fr - last_fr <= 5 and new_fr - last_fr >1:
            
            
            last_x = tracker.tracks[-1][2]
            last_y = tracker.tracks[-1][3]
            last_w = tracker.tracks[-1][4]
            last_h = tracker.tracks[-1][5]
            
            new_x = detection[2]
            new_y = detection[3]
            new_w = detection[4]
            new_h = detection[5]
            
            for frnum in range(int(last_fr)+1,int(new_fr)):
                new_det = np.ones((detection.shape[0],))*-1
                new_det[0] =frnum
                new_det[1] = tracker.id
                
                new_det[2] = last_x +((new_x - last_x)/(new_fr - last_fr))*(frnum - last_fr)
                new_det[3] = last_y +((new_y - last_y)/(new_fr - last_fr))*(frnum - last_fr)
                new_det[4] = last_w +((new_w - last_w)/(new_fr - last_fr))*(frnum - last_fr)
                new_det[5] = last_h +((new_h - last_h)/(new_fr - last_fr))*(frnum - last_fr)
                
                tracker.add_track(new_det)
                
                
def get_bb_content_cv2(detection, frame):
    img = Image.open(frame)

    x = detection[2]
    y = detection[3]
    width = detection[4]
    height = detection[5]
    bb_content = img.crop((  np.maximum(int(x),0), np.maximum(int(y),0) ,int(x) + math.ceil(width),int(y) + math.ceil(height)))
    resized = bb_content.resize((256,128),Image.NEAREST)
    return resized

def get_bb_content_pil(detection, frame):
    
 #   img = Image.open(frame)
    img =frame
    x = detection[2]
    y = detection[3]
    width = detection[4]
    height = detection[5]
    bb_content = img.crop((  np.maximum(int(x),0), np.maximum(int(y),0) ,int(x) + math.ceil(width),int(y) + math.ceil(height)))
    
    resized = bb_content.resize((256,128),Image.NEAREST)
    return resized


def get_bb_content_pil_opened(detection, frame,shape=(256,128)):
    x = detection[2]
    y = detection[3]
    width = detection[4]
    height = detection[5]
    transform = transforms.Compose([
     transforms.Resize(shape),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
    ])

    bb_content = frame.crop(
        (np.maximum(int(x), 0), np.maximum(int(y), 0), int(x) + math.ceil(width), int(y) + math.ceil(height)))
    return transform(bb_content)


def get_velocity(det_t_minus_1, det_t):
    x_t_1,y_t_1 = get_bb_center(det_t_minus_1)
    x_t,y_t = get_bb_center(det_t)

    return torch.Tensor([x_t - x_t_1, y_t - y_t_1])

def get_bb_center(detection):
    return (detection[2]) +(( detection[4] /2)), (detection[3])+ ((detection[5]/ 2))



def get_coordinates_in_grid(detection,frame_width, frame_height, grid_width, grid_height, subgrid_width, subgrid_height):

    cell_h = frame_height // grid_height
    cell_w = frame_width  // grid_width

    h_idx = int((detection[3]  + (detection[5] // 2)) // cell_h)
    w_idx = int((detection[2] + (detection[4]  // 2)) // cell_w)

    return h_idx, w_idx



def get_occupancy_grid(detection, detection_list, frame_width, frame_height, grid_width, grid_height, subgrid_width, subgrid_height):

    occupancy_grid = np.zeros((subgrid_height, subgrid_width), dtype=np.float)
    grid_center_x = int(subgrid_width // 2)
    grid_center_y = int(subgrid_height // 2)

    box_y, box_x = get_coordinates_in_grid(detection,frame_width,
                                                frame_height, grid_width, grid_height, subgrid_width, subgrid_height)
    
    for det in detection_list:
        neighbour_y, neighbour_x = get_coordinates_in_grid(det,frame_width,
                                                                frame_height, grid_width, grid_height,
                                                                subgrid_width, subgrid_height)

        delta_y = neighbour_y - box_y
        delta_x = neighbour_x - box_x

        if abs(delta_y) <= (subgrid_height // 2) and abs(delta_x) <= subgrid_width // 2:
            grid_y = grid_center_y + delta_y
            grid_x = grid_center_x + delta_x

            occupancy_grid[grid_y][grid_x] = 1.0

    return torch.Tensor(occupancy_grid.flatten())


def evaluate(gts,filename):
    
    mh = motmetrics.metrics.create()
    acc = motmetrics.MOTAccumulator(auto_id=True)
    threshold = 0.5
    detections = []    
    
    with open(filename ) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            detections.append(row)

    detections = np.array(detections)
    detections = detections.astype('float64')
    
    detections_by_frame = groupby(detections,0)
    gts_by_frame = groupby(gts,0)

    for frameid in gts_by_frame.keys():
        # print("frameid: ", int(frameid)+1)
        # get gt ids
        gt_bboxes = gts_by_frame[frameid]
        gt_ids = gt_bboxes[:, 1].astype(np.int32).tolist()

        if frameid in detections_by_frame.keys():
            id_track = detections_by_frame[frameid][:, 1].tolist()
            mask_IOU = np.zeros((len(detections_by_frame[frameid]), len(gts_by_frame[frameid])))
            distance_matrix = []
            for i, bbox in enumerate(detections_by_frame[frameid]):
                iou,_,_ = overlap(bbox, gt_bboxes)
                th = np.zeros_like(iou)
                th[np.where(iou <= threshold)] = 1.0
                mask_IOU[i, :] = th
                distance_matrix.append(1.0-iou)
            
            distance_matrix = np.vstack(distance_matrix)
            distance_matrix[np.where(mask_IOU == 1.0)] = np.nan

            acc.update(
                gt_ids,  # number of objects = matrix width
                id_track,  # number of hypothesis = matrix height
                np.transpose(distance_matrix)
            )
        else:
            acc.update(
                gt_ids,  # number of objects = matrix width
                [],      # number of hypothesis = matrix height
                [[], []]
            )

    summary = mh.compute(acc, metrics=['motp', 'mota', 'num_false_positives', 'num_misses', 'num_switches',
                                       'num_objects', 'num_matches','mostly_lost','mostly_tracked'], name='final')


    strsummary = motmetrics.io.render_summary(
        summary,
        formatters={'mota': '{:.2%}'.format},
        namemap={'motp': 'MOTP', 'mota': 'MOTA', 'num_false_positives': 'FP', 'num_misses': 'FN',
                 'num_switches': "ID_SW", 'num_objects': 'num_objects','ML':'mostly_lost', 'MT':'mostly_tracked' }
    )
    print(strsummary)
    
    return summary #print(summary['motp'])



def groupby(x,idx): 
    x_uniques = np.unique(x[:,idx]) 
    return {xi:x[x[:,idx]==xi] for xi in x_uniques } 


def write_results(targets,filename):
    all_tracks  = []
    for target in targets:
        all_tracks += target.tracks
        
    all_tracks = torch.stack(all_tracks).numpy()
    all_tracks_f = groupby(all_tracks,0)

    with open(filename,'w+') as file:
        writer = csv.writer(file, delimiter=',')
        for frame in all_tracks_f.keys():
            for track in all_tracks_f[frame]:
                row = [track[i] for i in range(track.shape[0])]    
                writer.writerow(row)



