import os
from itertools import chain
from PIL import Image
import random
import pickle
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import namedtuple, defaultdict
import warnings
import torch
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as tr_F
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)

from multiprocessing import Pool

import copy
import torch.nn as nn
import data_utils.dataset_utils as dataset_utils

from data_utils.dataset_utils import Box


def create_sequence_det_boxes_dict(det_boxes):
    det_boxes_dict = defaultdict(list)

    for det_box in det_boxes:
        det_boxes_dict[det_box.frame_number].append(det_box)

    det_boxes_dict = dict(det_boxes_dict)

    return det_boxes_dict

def fit_bbox_to_frame(bbox, frame):
    x, y, w, h = bbox
    fh, fw, _ = frame.shape
    w = min(w, fw-x)
    h = min(h, fh-y)
    bbox = np.array([x, y, w, h])
    return bbox

def get_cuhk_image_path(box, cuhk_images_path):
    image_path = os.path.join(
        cuhk_images_path,
        "{}_{}.jpg".format(str(box.obj_id).zfill(4), str(box.frame_number).zfill(2))
    )
    return image_path

class MOT_data(Dataset):
    """2DMOT2015 dataset."""

    def __init__(
            self,
            transform=None,
            crops=False,
            cuhk =False,
            velocities=False,
            occupancy_grids=False,
            balanced_batch=True,
            sequence_length=6,
            future_seq_len=6,
            full_image_grid_dim=15,
            occupancy_grid_dim=7,
            input_w=224,
            input_h=224,
            data_mot = '15',
            test_set=False,
            device = 'cpu',
            video_names = None,
    ):
        
        self.obj_id_counter = defaultdict(lambda: defaultdict(int))
        self.obj_id_frame_counter_neg = defaultdict(dict)
        self.obj_id_frame_counter_pos = defaultdict(dict)
        self.gap = 10
        self.limit = 200
        self.limit_ex_per_id = False
        
        
       
        self.dataset_path = "/media/data/gauthier/MOT"+data_mot+"/train/"
        self.only_pairs = True
        self.device = device
        self.all_futures = False
        if self.all_futures:
            assert not self.only_pairs
            assert not self.limit_ex_per_id

        print("limit_ex {} limit {} gap {}".format(self.limit_ex_per_id, self.limit, self.gap))
        print("only pairs: {}".format(self.only_pairs))
        print("all futures: {}".format(self.all_futures))

        self.cuhk = cuhk
        self.test_set = False
        self.crops = crops
        self.velocities = velocities
        self.occupancy_grids = occupancy_grids
        self.overlap_occ = 0.5  # original 0.7
        self.overlap_pos = 0.5

        self.future_seq_len = future_seq_len
        
        self.gt_id =0
        self.max_iou_detections_dict = dict()

        self.input_w, self.input_h = input_w, input_h
        self.margin = 0
        self.sequence_length = sequence_length
        self.max_sequence_length = sequence_length
        self.full_image_grid_dim = full_image_grid_dim
       
        self.occupancy_grid_dim = occupancy_grid_dim

        self.transform = transform

        print("dataset_type: ",data_mot,  "train")
        
        if video_names is None:
            if data_mot == '15':
                self.video_names = [
                   'ADL-Rundle-6',
                   'ETH-Bahnhof',
                   'KITTI-13',
                   'TUD-Stadtmitte',
                    "ETH-Pedcross2",
                    "ADL-Rundle-8",
                    "Venice-2",
                    "KITTI-17",
                    'PETS09-S2L1'
                ]       
            elif data_mot =='16':
                self.video_names = [
                   'MOT16-13',
                   'MOT16-11',
                   'MOT16-10',
                   'MOT16-09',
                    "MOT16-05",
                    "MOT16-04",
                    "MOT16-02"
                ]

            elif data_mot == '17' :
                self.video_names = [
                   'MOT17-02-FRCNN',
                   'MOT17-04-DPM',
                   'MOT17-05-DPM',
                   'MOT17-09-DPM',
                    "MOT17-09-SDP",
                    "MOT17-10-SDP",
                    "MOT17-11-SDP",
                    "MOT17-13-SDP",
                    "MOT17-02-SDP",
                    "MOT17-04-SDP",
                    'MOT17-05-FRCNN',
                    "MOT17-09-FRCNN",
                    'MOT17-10-FRCNN',
                    'MOT17-11-FRCNN',
                    'MOT17-13-DPM'
                ]
            elif data_mot == '15_16':
                self.video_names = [
                    'TUD-Campus',
                    'ETH-Bahnhof',
                ]


            else:
                print('wrong data_mot')
        else:
            self.video_names = video_names
        self.gt_boxes_dict = {}
        self.det_boxes_dict = {}
        self.total_gt_boxes = []
        self.biggest_gt_id =0
        for video_name in self.video_names:
            print(self.dataset_path+ video_name)

            frame_h, frame_w, _ = cv2.imread(self.dataset_path+ video_name+ '/img1/000011.jpg').shape
            det_path = self.dataset_path+ video_name+ '/det/det.txt'
            det_boxes = pd.read_csv(det_path, header=None).values[:, :7]
            det_boxes = dataset_utils.parse_boxes(det_boxes, video_name, frame_h, frame_w)
            
            det_boxes = [b for b in det_boxes if (b.conf > 10.0 and data_mot=='15' ) or (b.conf > 0.05 and data_mot=='16') or  (b.conf > 0.1 and data_mot=='17')]

            self.det_boxes_dict[video_name] = create_sequence_det_boxes_dict(det_boxes)

            gt_path =self.dataset_path+ video_name+ '/gt/gt.txt'
            gt_boxes = pd.read_csv(gt_path, header=None).values[:, :7]
            gt_boxes = gt_boxes[ gt_boxes[:,-1]>0.5]
            max_id = np.max(copy.deepcopy(gt_boxes[:,1]))
            gt_boxes[:,1] = gt_boxes[:,1] + self.biggest_gt_id
            self.biggest_gt_id += max_id
            gt_boxes = dataset_utils.parse_boxes(gt_boxes, video_name, frame_h, frame_w)
            self.gt_boxes_dict[video_name] = dataset_utils.create_sequence_gt_boxes_dict(gt_boxes)
            self.total_gt_boxes.extend(gt_boxes)
        print("gt boxes dict created size: {}".format(len(self.total_gt_boxes)))
        
        """
        if self.occupancy_grids:
            gt_boxes_hash = dataset_utils.calculate_hash([
                'gt_data',
                self.total_gt_boxes,
                self.occupancy_grid_dim,
                self.full_image_grid_dim,
                self.dataset_path,
                self.limit,
                self.limit_ex_per_id,
                self.gap
            ])

            cache_path = os.path.join("../cache", gt_boxes_hash+".pickle")

            if os.path.isfile(cache_path):
                with open(cache_path, "rb") as cache_file:
                    print("loading from cache: ", cache_path)
                    self.occupancy_grids_dict = pickle.load(cache_file)
                    print("loaded")
            else:
                print("creating occupancy grids")
                
                self.occupancy_grids_dict = {
                    gt_box: self.create_occupancy_grid(
                        gt_box,
                        self.occupancy_grid_dim,
                        self.full_image_grid_dim,
                        i
                    ) for i, (gt_box) in enumerate(self.total_gt_boxes) }
                print("occupancy grids created")
                print("caching")
                with open(cache_path, "wb+") as cache_file:
                    pickle.dump(self.occupancy_grids_dict, cache_file)
                print("cached to: ", cache_path)

        else:
            self.occupancy_grids_dict = None
        total_gt_boxes_hash = dataset_utils.calculate_hash(
            self.total_gt_boxes+det_boxes+['gt_data',
                self.velocities, self.crops,self.video_names, self.occupancy_grids, self.sequence_length,
                self.limit_ex_per_id, self.limit, self.gap, self.only_pairs
            ]
        )
        """
        self.gt_boxes_sequences = self.create_sequences(self.total_gt_boxes)

            
            
        
        if self.cuhk :
            cuhk_sequences = self.create_cuhk_sequences()
            
            print('using cuhk dataset')
            print('size of cuhk sequences',len(cuhk_sequences))
            self.gt_boxes_sequences.extend(cuhk_sequences)



        labels = [s['label'] for s in self.gt_boxes_sequences]

        print("sequences ", len(self.gt_boxes_sequences))
        pos = sum(labels)
        neg = len(labels) - pos
        print("positive samples: {} negative samples: {}".format(pos, neg))

        from collections import Counter
        c = Counter([s['sequence_boxes'].__repr__() for s in self.gt_boxes_sequences])
        print("duplicate sequence boxes:", sum([c == 2 for c in c.values()]))
        print("single sequence boxes:", sum([c == 1 for c in c.values()]))

        duplicate_labels = [s['label'] for s in self.gt_boxes_sequences if c[s['sequence_boxes'].__repr__()] == 2]
        print("duplicate labels:", Counter(duplicate_labels))

        duplicate_labels = [s['label'] for s in self.gt_boxes_sequences if c[s['sequence_boxes'].__repr__()] == 1]
        print("single labels:", Counter(duplicate_labels))

        if self.crops:
            num_diff_obj = len(set(
                [str(s['candidate_box'].obj_id)+str(s['candidate_box'].video_name) for s in self.gt_boxes_sequences]
            ))
            print("num diff obj: ", num_diff_obj)


        print("{} sequences created".format(len(self.gt_boxes_sequences)))

        num_of_positive_examples = sum([b['label'] for b in self.gt_boxes_sequences])
        num_of_negative_examples = len(self.gt_boxes_sequences)-num_of_positive_examples
        self._class_weights = [num_of_negative_examples, num_of_positive_examples]


    
    def class_weights(self):
        return self._class_weights

    def get_coordinates_in_grid(self,box, full_image_grid_dim):
        def get_image_path(box):
            img_path =self.dataset_path+box.video_name+ '/img1/'+str(int(box.frame_number)).zfill(6) + ".jpg"
            return img_path

        img = cv2.imread(get_image_path(box))
        height, width, _ = img.shape

        cell_h = height // full_image_grid_dim
        cell_w = width  // full_image_grid_dim

        h_idx = int((box.bb_top + (box.bb_height // 2)) // cell_h)
        w_idx = int((box.bb_left + (box.bb_width // 2)) // cell_w)

        return h_idx, w_idx
        
        
    def create_occupancy_grid(self, box, occupancy_grid_dim, full_image_grid_dim,  i=None):
        if i%100 == 0 and i!=0:
            print(i)
            
            
        occupancy_grid = np.zeros((occupancy_grid_dim, occupancy_grid_dim), dtype=np.float)
        grid_center_x = int(occupancy_grid_dim // 2)
        grid_center_y = int(occupancy_grid_dim // 2)
        box_y, box_x = self.get_coordinates_in_grid(box, full_image_grid_dim)
        
        for _,neighbour in self.gt_boxes_dict[box.video_name][box.frame_number].items():
            
            neighbour_y, neighbour_x = self.get_coordinates_in_grid(neighbour, full_image_grid_dim)
            delta_y = neighbour_y - box_y
            delta_x = neighbour_x - box_x

            if abs(delta_y) <= (occupancy_grid_dim // 2) and abs(delta_x) <= occupancy_grid_dim // 2:
                # neighbour is inside the occupancy grid
                grid_y = grid_center_y + delta_y
                grid_x = grid_center_x + delta_x
                occupancy_grid[grid_y][grid_x] = 1.0

        return occupancy_grid.flatten()
        
    def check_occluded(self, box):
        frame_detections = self.gt_boxes_dict[box.video_name][box.frame_number]
        if len(frame_detections) == 1: return False

        # assumption that the object is always occluded by one object, 
        occluded_pecentage = max([
            dataset_utils.box_intersection_over_area(box, b)
            for b in frame_detections.values() if box.obj_id != b.obj_id
        ])
        return occluded_pecentage > self.overlap_occ

    def find_max_iou_detection(self, gt_box):
        res = self.max_iou_detections_dict.get(gt_box)
        if res is None:
            # find max detections and insert to dict
                
            detections = self.det_boxes_dict[gt_box.video_name].get(gt_box.frame_number, [])
            if len(detections) > 0:
                iou_s = np.array([dataset_utils.box_intersection_over_union(
                    det_box,
                    gt_box
                ) for det_box in detections])

                max_detection_ind = np.argmax(iou_s)

                self.max_iou_detections_dict[gt_box] = detections[max_detection_ind], iou_s[max_detection_ind]

            else:
                self.max_iou_detections_dict[gt_box] = None, 0

        return self.max_iou_detections_dict.get(gt_box)

    
    def create_cuhk_sequences(self):
        self.cuhk_images_path =  '/media/data/gauthier/CUHK03_dataset/detected/train_resized/'
        ids = set([n[:4] for n in os.listdir(self.cuhk_images_path)])
        cuhk_sequences = []
        for id in ids:
            def create_samples_from_camera_boxes(camera_boxes):
                # returns 4 samples 2 positive and 2 negative
                # one positive sample is created by taking the last camera box as candidate and others as sequences, the
                # other positive sample is created by taking the first camera box as candidate and others as sequences
                # and then negative samples are created for each of the two positive samples
                samples = []

                def get_first_negative_sample(neg_id):
                    frame_number = 0
                    while True:
                        negative_candidate_box = Box(
                            video_name='cuhk', frame_number=frame_number, obj_id=neg_id,
                            bb_left=0, bb_top=0, bb_width=self.input_w, bb_height=self.input_h,
                            conf=1.0
                        )
                        assert frame_number < 10
                        if get_cuhk_image_path(negative_candidate_box, self.cuhk_images_path):
                            return negative_candidate_box
                        else:
                            frame_number += 1

                if len(camera_boxes) < 2:
                    return []

                other_ids = list(range(1, int(id))) + list(range(int(id) + 1, len(ids)))
                neg_ids = [i.zfill(4) for i in map(str, random.sample(other_ids, len(camera_boxes)))]

                for cand_ind in range(len(camera_boxes)):
                    # sample candidate first
                    sequence_boxes = camera_boxes[:cand_ind]+camera_boxes[cand_ind+1:]

                    if self.sequence_length >= len(sequence_boxes):
                        pad_size = self.sequence_length - len(sequence_boxes)
                        sequence_boxes = [sequence_boxes[0]]*pad_size + sequence_boxes

                    else:
                        sequence_boxes = sequence_boxes[:self.sequence_length]

                    positive_candidate_box = camera_boxes[cand_ind]
                    negative_candidate_box = get_first_negative_sample(neg_ids[cand_ind])
                    
                    positive_candidate_box = MOT_data.box_like_with_id(positive_candidate_box,
                                                              int(positive_candidate_box.obj_id) + self.biggest_gt_id)
                    
                    sequence_boxes[-1] = MOT_data.box_like_with_id(sequence_boxes[-1],
                                                                   int(sequence_boxes[-1].obj_id)+ self.biggest_gt_id)
                    
                    negative_candidate_box = MOT_data.box_like_with_id(negative_candidate_box,
                                                              int(negative_candidate_box.obj_id) + self.biggest_gt_id)
                    samples.append(self.create_sample(
                        box_gt = sequence_boxes[-1],candidate_gt = positive_candidate_box,
                        candidate_box=positive_candidate_box,
                        candidate_label=1,
                        sequence_boxes=sequence_boxes,
                        sequence_velocities=None,
                        sequence_locations=None,
                        sequence_frame_numbers=None,
                        sequence_occupancy_grids=None))

                    samples.append(self.create_sample(
                        box_gt = sequence_boxes[-1],candidate_gt = negative_candidate_box,
                        candidate_box=negative_candidate_box,
                        candidate_label=0,
                        sequence_boxes=sequence_boxes,
                        sequence_velocities=None,
                        sequence_locations=None,
                        sequence_frame_numbers=None,
                        sequence_occupancy_grids=None))


                return samples

            first_camera_boxes = [
                Box(video_name='cuhk', frame_number=i, obj_id=(id),
                    bb_left=0, bb_top=0, bb_width=self.input_w, bb_height=self.input_h,
                    conf=1.0
                    ) for i in range(5)
            ]
            first_camera_boxes = [
                b for b in first_camera_boxes if os.path.isfile(get_cuhk_image_path(b, self.cuhk_images_path))
            ]

            cuhk_sequences.extend(create_samples_from_camera_boxes(first_camera_boxes))

            second_camera_boxes = [
                Box(video_name='cuhk', frame_number=i, obj_id=(id),
                    bb_left=0, bb_top=0, bb_width=self.input_w, bb_height=self.input_h,
                    conf=1.0
                    ) for i in range(5, 10)
            ]
            second_camera_boxes = [
               b for b in second_camera_boxes if os.path.isfile(get_cuhk_image_path(b, self.cuhk_images_path))
            ]
            cuhk_sequences.extend(create_samples_from_camera_boxes(second_camera_boxes))

        return cuhk_sequences

    
    def modify_box(self,box):
        #mean then std
        shape_mod = np.random.gumbel(0.9,0.1 )
        new_width = box.bb_width*shape_mod
        new_height = box.bb_height*shape_mod
        
        x_rand_offset = np.random.gumbel(0,0.1 )#*(np.random.uniform(low=0, high=0.5))
        
        y_rand_offset = np.random.gumbel(0,0.1 )#*(np.random.uniform(low=0, high=0.5))

        new_x = box.bb_left+ box.bb_width*x_rand_offset
        new_y = box.bb_top+ box.bb_height*y_rand_offset
        
        
        return Box(
            video_name=box.video_name, frame_number=box.frame_number, obj_id=box.obj_id,
            bb_left=new_x, bb_top=new_y, bb_width=new_width, bb_height=new_height,
            conf=box.conf
        )

    def create_sequences(self, gt_boxes):
        sequences_ = []
        for i, gt_box in enumerate(gt_boxes):
            if i % 1000 == 0: print("create sequences step {}/{} video: {} ".format(i, len(gt_boxes), gt_box.video_name))

            if self.velocities:

                if gt_box.frame_number-1 not in self.gt_boxes_dict[gt_box.video_name]: continue
                

                if gt_box.obj_id not in self.gt_boxes_dict[gt_box.video_name][gt_box.frame_number-1]: continue

            if any([gt_box.obj_id not in self.gt_boxes_dict[gt_box.video_name].get(gt_box.frame_number+i, []) for i in range(self.sequence_length)]):
                continue

            if any([self.check_occluded(self.gt_boxes_dict[gt_box.video_name][gt_box.frame_number+i][gt_box.obj_id])
                   for i in range(self.sequence_length)]):
                continue

            sequence_boxes = [
                    self.gt_boxes_dict[gt_box.video_name][gt_box.frame_number+i][gt_box.obj_id]
                              for i in range(self.sequence_length)
            ]


            sequence_boxes = [self.modify_box(box) for box in sequence_boxes] 
            pos_s = []
            neg_s = []

            if self.velocities:
                first_sequence_box = sequence_boxes[0]
                pre_sequence_box = self.modify_box( self.gt_boxes_dict[first_sequence_box.video_name][first_sequence_box.frame_number - 1][gt_box.obj_id])
              

                all_seq_boxes = [pre_sequence_box] + sequence_boxes
                all_seq_centers = dataset_utils.calc_centers(
                    np.array([dataset_utils.box_to_bbox(b) for b in all_seq_boxes]))

                # locations
                sequence_locations = [l for l in all_seq_centers]

                # frame_numbers
                sequence_frame_numbers = [b.frame_number for b in all_seq_boxes]

                # velocities
                sequence_velocities = all_seq_centers[1:] - all_seq_centers[:-1]
                sequence_velocities = [v for v in sequence_velocities]

            if self.occupancy_grids:
                sequence_occupancy_grids = [self.create_occupancy_grid(box,
                        self.occupancy_grid_dim,
                        self.full_image_grid_dim,i
                    ) for box in sequence_boxes]

            following_sequence = [
                self.gt_boxes_dict[gt_box.video_name][gt_box.frame_number + i] for i in
                                          range(self.sequence_length, self.sequence_length + self.future_seq_len)
                if gt_box.frame_number+i in self.gt_boxes_dict[gt_box.video_name]
            ]

            positive_candidate_boxes_gt = [fr[gt_box.obj_id] for fr in following_sequence if gt_box.obj_id in fr]
            positive_candidate_boxes_gt = [b for b in positive_candidate_boxes_gt if not self.check_occluded(b)]

            # random pick one
            if len(positive_candidate_boxes_gt) > 1:
                positive_candidate_boxes_gt = [random.choice(positive_candidate_boxes_gt)]


            for positive_candidate_box_gt in  positive_candidate_boxes_gt:

                sample = self.create_sample(box_gt = gt_box,candidate_gt = positive_candidate_box_gt,
                                            candidate_box=positive_candidate_box_gt, candidate_label=1,
                                            sequence_boxes=sequence_boxes,
                                            sequence_velocities=sequence_velocities if self.velocities else None,
                                            sequence_locations=sequence_locations if self.velocities else None,
                                            sequence_frame_numbers=sequence_frame_numbers if self.velocities else None,
                                            sequence_occupancy_grids=sequence_occupancy_grids if self.occupancy_grids else None)
                                     
                positive_sample = sample

                if self.test_set or not self.limit_ex_per_id:
                    pos_s.append(positive_sample)

                elif self.obj_id_counter[gt_box.video_name][gt_box.obj_id] < self.limit:
                    last_obj_frame = self.obj_id_frame_counter_pos[positive_candidate_box.video_name].get(positive_candidate_box.obj_id, None)
                    
                    if last_obj_frame is None or sequence_boxes[0].frame_number-last_obj_frame > self.gap:
                        pos_s.append(positive_sample)

            # find and add negative samples
            negative_sample_candidates = []
            for offset in range(self.future_seq_len):
                # we take candidates from future_seq_len frames in the future
                if gt_box.frame_number+self.sequence_length+offset not in self.gt_boxes_dict[gt_box.video_name]:
                    continue

                candidates = self.gt_boxes_dict[gt_box.video_name][gt_box.frame_number+self.sequence_length+offset].values()

                candidates = [cand for cand in candidates if not self.check_occluded(cand)]

                # leave only the candidates with corresponding detections

                if len(candidates) > 0:
                    negative_candidate_box_gt = MOT_data.find_random_negative_candidate_box(candidates, gt_box)

                    if negative_candidate_box_gt is not None:

                        negative_sample = self.create_sample(
                            box_gt = gt_box,candidate_gt = negative_candidate_box_gt,
                            candidate_box=negative_candidate_box_gt,
                            candidate_label=0,
                            sequence_boxes=sequence_boxes,
                            sequence_velocities=sequence_velocities if self.velocities else None,
                            sequence_locations=sequence_locations if self.velocities else None,
                            sequence_frame_numbers=sequence_frame_numbers if self.velocities else None,
                            sequence_occupancy_grids=sequence_occupancy_grids if self.occupancy_grids else None
                         #   sequence_crops = sequence_crops if self.crops else None
                        )

                        negative_sample_candidates.append(negative_sample)

            # random pick one
            if len(negative_sample_candidates) > 0:
                if self.test_set or self.all_futures:
                    for sam in negative_sample_candidates:
                        neg_s.append(sam)
                else:
                    neg_sample = random.choice(negative_sample_candidates)

                    if not self.limit_ex_per_id:
                        neg_s.append(neg_sample)

                    elif self.obj_id_counter[gt_box.video_name][gt_box.obj_id] < self.limit:

                        last_obj_frame = self.obj_id_frame_counter_neg[gt_box.video_name].get(
                            gt_box.obj_id, None)

                        if last_obj_frame is None or neg_sample["sequence_boxes"][0].frame_number-last_obj_frame > self.gap:
                            neg_s.append(neg_sample)

            if not self.test_set and not self.all_futures:
                if self.only_pairs:
                    if len(pos_s) != 1 or len(neg_s) != 1:
                        pos_s, neg_s = [], []
                assert len(pos_s) <= 1 and len(neg_s) <= 1

            if not self.test_set and self.limit_ex_per_id and not self.all_futures:
                if len(pos_s) > 0 and len(neg_s) > 0:
                    assert len(pos_s) == 1
                    assert len(neg_s) == 1
                    self.obj_id_counter[gt_box.video_name][gt_box.obj_id]+=1
                    self.obj_id_frame_counter_pos[gt_box.video_name][gt_box.obj_id]=pos_s[0]['candidate_box'].frame_number
                    self.obj_id_counter[gt_box.video_name][gt_box.obj_id] += 1
                    self.obj_id_frame_counter_neg[gt_box.video_name][gt_box.obj_id]=neg_s[0]["candidate_box"].frame_number

            sequences_.extend(pos_s)
            sequences_.extend(neg_s)

            if not self.test_set and self.limit_ex_per_id:
                assert len(sequences_) == sum(
                    chain(*[
                        obj_c.values() for obj_c in self.obj_id_counter.values()
                    ]))

        return sequences_

    @staticmethod
    def box_like_with_id(box, id):
        box_list = list(box)
        box_list[2] = int(id)
        new_box = Box(*box_list)
        return new_box


    @staticmethod
    def find_random_negative_candidate_box(candidates, gt_box):
        for candidate_box in candidates:
            if gt_box.obj_id != candidate_box.obj_id and type(candidate_box) is not list:
                return candidate_box
        else:
            return None


    def create_sample(self,box_gt,candidate_gt, candidate_box, candidate_label, sequence_boxes=None, sequence_velocities=None, sequence_locations=None, sequence_frame_numbers=None, sequence_occupancy_grids=None,sequence_crops = None):
        if sequence_boxes[-1].obj_id != -1:
            assert candidate_label == int(candidate_box.obj_id == sequence_boxes[-1].obj_id)

        if type(candidate_box) is not dataset_utils.Box:
            raise ValueError("Candidate box must be of type Box")

        if candidate_label not in [0, 1]:
            raise ValueError("Candidate label must be 0 or 1")

        sample = {
            "label": candidate_label
        }
        
        candidate_box = self.modify_box(candidate_box)
        box_gt  = self.modify_box(box_gt)
        
        if sequence_boxes is None:
            raise ValueError("sequcence_boxes can't be None if self.crops is True")

        sample["sequence_boxes"] = sequence_boxes
        sample["candidate_box"] = candidate_box
        
        sample["box_gt"] = box_gt
        sample["candidate_gt"] = candidate_gt
        
        if self.velocities:
            if sequence_velocities is None:
                raise ValueError("sequcence_velocities can't be None if self.velocities is True")

            if sequence_boxes is None:
                raise ValueError("sequcence_boxes can't be None if self.velocities is True")

            cand_center = dataset_utils.box_center(candidate_box)
            # locations
            sample["sequence_locations"] = sequence_locations  # sequence_len + 1
            sample["candidate_location"] = cand_center

            # velocities
            sample["sequence_velocities"] = sequence_velocities

            last_seq_center = dataset_utils.box_center(sequence_boxes[-1])
            sample["candidate_velocity"] = cand_center-last_seq_center

            # frame numbers
            sample["sequence_frame_numbers"] = sequence_frame_numbers  # sequence_len + 1
            sample["candidate_frame_number"] = candidate_box.frame_number

        if self.occupancy_grids:
            if sequence_occupancy_grids is None:
                raise ValueError("sequence_occupancy_grids can't be None if self.occupancy_grids is True")
    
            sample["sequence_occupancy_grids"] = sequence_occupancy_grids

            sample["candidate_occupancy_grid"] = self.create_occupancy_grid(candidate_box,self.occupancy_grid_dim,
                                                                            self.full_image_grid_dim,0
                                                                            )
        return sample

    def __len__(self):
        return len(self.gt_boxes_sequences)



    
    def crop_and_resize_box_from_frame(self, box):
        if box.video_name == "cuhk":
            ##cuhk are also of the good size
            box = MOT_data.box_like_with_id(box,box.obj_id - self.biggest_gt_id)
            image_path = get_cuhk_image_path(box, self.cuhk_images_path)
            image = Image.open(image_path)
            
            assert image is not None
            return image

        else:
            def get_image_path(box, dataset_path):
                img_path = self.dataset_path+box.video_name+ '/img1/'+str(int(box.frame_number)).zfill(6) + ".jpg"
                return img_path

            img_name = get_image_path(box, self.dataset_path)
            
            img = Image.open(img_name)
            bb_content = img.crop((  np.maximum(int(box.bb_left )-self.margin,0),
                                   np.maximum(int(box.bb_top )-self.margin,0) ,
                                   int(box.bb_left) + math.ceil(box.bb_width)+ box.bb_width  + 2*self.margin,
                                   int(box.bb_top) + math.ceil(box.bb_height) + 2*self.margin))
            
            return bb_content

    
    def __getitem__(self, idx):
        # index of the last box to be fed into the LSTM
        sample = {"label": self.gt_boxes_sequences[idx]['label']}

        boxes = self.gt_boxes_sequences[idx]["sequence_boxes"]
        candidate_box = self.gt_boxes_sequences[idx]["candidate_box"]
        sample["sequence_boxes"]= boxes
        sample["candidate_box"]= candidate_box
        
        sample["box_gt"] = self.gt_boxes_sequences[idx]["box_gt"]
        sample["candidate_gt"] = self.gt_boxes_sequences[idx]["candidate_gt"]

        if self.crops:
            # crop boxes from frames and reshape to fixes size
            preprocessed_boxes = []
            for i, box in enumerate(boxes):
                image = self.crop_and_resize_box_from_frame(box)
                preprocessed_boxes.append(image)
                
            candidate_crop = self.crop_and_resize_box_from_frame(candidate_box)
            sample["image_crops"] = preprocessed_boxes
            sample["candidate_crop"] = candidate_crop
        
        if self.velocities:
            sample["sequence_velocities"] = self.gt_boxes_sequences[idx]["sequence_velocities"]
            sample["candidate_velocity"] = self.gt_boxes_sequences[idx]["candidate_velocity"]
            
            sample["sequence_locations"] = self.gt_boxes_sequences[idx]["sequence_locations"]
            sample["candidate_location"] = self.gt_boxes_sequences[idx]["candidate_location"]

            sample["sequence_frame_numbers"] = self.gt_boxes_sequences[idx]["sequence_frame_numbers"]
            sample["candidate_frame_number"] = self.gt_boxes_sequences[idx]["candidate_frame_number"]

        if self.occupancy_grids:
            sample["sequence_occupancy_grids"] = self.gt_boxes_sequences[idx]["sequence_occupancy_grids"]
            sample["candidate_occupancy_grid"] = self.gt_boxes_sequences[idx]["candidate_occupancy_grid"]
            
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, crops=True, velocities=True, occupancy_grids=True, augment=False,shape=(224,224)):
        self.crops = crops
        self.velocities = velocities
        self.occupancy_grids = occupancy_grids

        self.toTen = transforms.ToTensor()
        self.toPil = transforms.ToPILImage()
        self.augment  = augment
        self.shape = shape
        
        
    def __call__(self, sample):
        
        if self.crops:
            image_crops, candidate_crop = sample['image_crops'], sample['candidate_crop']

            transform_norm = transforms.Compose([
                 transforms.Resize(self.shape),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

            image_crops = [
                transform_norm(image_crop)
                for  image_crop in image_crops
            ]
            candidate_crop = transform_norm(candidate_crop)

            sample["image_crops"] = image_crops
            sample["candidate_crop"] = candidate_crop

        
        sample["label"] = torch.tensor(sample["label"]).float()
        
        sample["sequence_boxes"] = sample["sequence_boxes"]
        sample["candidate_box"] = sample["candidate_box"]

        if self.velocities:
            # velocities
            sequence_velocities, candidate_velocity = sample['sequence_velocities'], sample['candidate_velocity']
            sequence_velocities = [
                torch.from_numpy(velocities).float()
                for i, velocities in enumerate(sequence_velocities)
            ]
            candidate_velocity = torch.from_numpy(candidate_velocity).float()

            sample["sequence_velocities"] = sequence_velocities
            sample["candidate_velocity"] = candidate_velocity

            # locations
            sequence_locations, candidate_location = sample['sequence_locations'], sample['candidate_location']
            sequence_locations = [
                torch.from_numpy(locations).float()
                for i, locations in enumerate(sequence_locations)
            ]
            candidate_location = torch.from_numpy(candidate_location).float()

            sample["sequence_locations"] = sequence_locations
            sample["candidate_location"] = candidate_location


            sequence_frame_numbers, candidate_frame_number = sample['sequence_frame_numbers'], sample['candidate_frame_number']
            sequence_frame_numbers = torch.tensor(sequence_frame_numbers)
            candidate_frame_number = torch.tensor(candidate_frame_number)

            sample["sequence_frame_numbers"] = sequence_frame_numbers
            sample["candidate_frame_number"] = candidate_frame_number

        if self.occupancy_grids:
            sequence_occupancy_grids, candidate_occupancy_grid = sample['sequence_occupancy_grids'], sample['candidate_occupancy_grid']
            sequence_occupancy_grids = [
                torch.from_numpy(occupancy_grid).float()
                for i, occupancy_grid in enumerate(sequence_occupancy_grids)
            ]
            candidate_velocity = torch.from_numpy(candidate_occupancy_grid).float()

            sample["sequence_occupancy_grids"] = sequence_occupancy_grids
            sample["candidate_occupancy_grid"] = candidate_velocity

        return sample


def get_loaders(
        batch_size,
        shuffle=False,
        cuhk=False,
        crops=False,
        velocities=False,
        boxes=False,
        augment =False,
        occupancy_grids=True,
        future_seq_len=6,
        num_workers=0,
        train=True,
        test=False,
        crop_shape =(256,128),
        data_mot= '17',
        train_sequence_length=6,
        test_sequence_length=6,
        device = 'cpu',
        video_names= None
):
    composed = transforms.Compose([ToTensor(crops=crops,
                                            velocities=velocities, occupancy_grids=occupancy_grids,augment=augment,shape=crop_shape)])
    train_dataset = MOT_data(composed,cuhk=cuhk, crops=crops, velocities=velocities, occupancy_grids=occupancy_grids, test_set=False, sequence_length=train_sequence_length, future_seq_len=future_seq_len,data_mot=data_mot,device = device,
                            video_names=video_names)
    
    
    
    train_ld = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    train_ds, valid_ds = torch.utils.data.random_split(train_ld.dataset, (math.floor((9*len(train_ld.dataset))/10), math.ceil(len(train_ld.dataset)/10)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader
