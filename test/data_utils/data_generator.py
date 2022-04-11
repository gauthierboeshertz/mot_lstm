import numpy as np
from utils import overlap
from PIL import Image
import csv
import pandas as pd
from utils import get_bb_content_pil, get_velocity, get_occupancy_grid
import os
import cv2
import random
import copy
import torch
def txt_to_npy(filename):
    'file name without the .txt extention please'

    # read_file = pd.read_fwf(filename+'.txt')
    # read_file.to_csv(filename+'.csv', index=None)
    with open(filename + '.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open(filename + '.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)

    with open(filename + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        frames = []

        for row in csv_reader:
            frames.append(row)

        npframes = np.array(frames)
        npframes = npframes.astype('float64')

        np.save(filename + '.npy', npframes)
        return npframes



def gen_mot2015_npy(path = '/media/data/gauthier'):
    detections_list = []
    frames_list = []
    gts_list = []
    for direc in os.listdir(path+'/MOT15/train/'):
        if direc == '.DS_Store' or direc[0:3]=='MOT':
            continue
        detections_list.append(txt_to_npy(path+'/MOT15/train/' + direc + '/det/det'))

        pic_list = []
        for pic in sorted(os.listdir(path+'/MOT15/train/' + direc + '/img1')):
            pic_list.append(path+'/MOT15/train/' + direc + '/img1/' + pic)
        frames_list.append(pic_list)

        gts_list.append(txt_to_npy(path+'/MOT15/train/' + direc + '/gt/gt'))
    return detections_list, gts_list, frames_list


def gen_mot2016_npy(path = '/media/data/gauthier'):
    detections_list = []
    frames_list = []
    gts_list = []
    for direc in os.listdir(path+'/MOT16/train/'):
        if direc == '.DS_Store':
            continue
        detections_list.append(txt_to_npy(path+'/MOT16/train/' + direc + '/det/det'))

        pic_list = []
        for pic in sorted(os.listdir(path+'/MOT16/train/' + direc + '/img1')):
            pic_list.append(path+'/MOT16/train/' + direc + '/img1/' + pic)
        frames_list.append(pic_list)

        gts_list.append(txt_to_npy(path+'/MOT16/train/' + direc + '/gt/gt'))
    return detections_list, gts_list, frames_list

def gen_mot2017_npy(path = '/media/data/gauthier'):
    detections_list = []
    frames_list = []
    gts_list = []
    for direc in os.listdir(path+'/MOT17/train/'):
        if direc =='.DS_Store':
            continue
            
            
        detections_list.append(path+'/MOT17/train/' + direc + '/det/det')

        pic_list = []
        for pic in sorted(os.listdir(path+'/MOT17/train/' + direc + '/img1')):
            pic_list.append(path+'/MOT17/train/' + direc + '/img1/' + pic)
        frames_list.append(pic_list)

        gts_list.append(txt_to_npy(path+'/MOT17/train/' + direc + '/gt/gt'))
    return detections_list, gts_list, frames_list


def gen_mot2015test_npy(path = '/media/data/gauthier'):
    detections_list = []
    frames_list = []
    gts_list = []
    for direc in os.listdir(path+'/MOT15/test/'):
        if direc == '.DS_Store':
            continue
        detections_list.append(txt_to_npy(path+'/MOT15/test/' + direc + '/det/det'))

        pic_list = []
        for pic in sorted(os.listdir(path+'/MOT15/test/' + direc + '/img1')):
            pic_list.append(path+'/MOT15/test/' + direc + '/img1/' + pic)
        frames_list.append(pic_list)

        #gts_list.append(txt_to_npy('data/2DMOT2015/train/' + direc + '/gt/gt'))
    return detections_list, frames_list

def gen_mot2016test_npy(path = '/media/data/gauthier'):
    detections_list = []
    frames_list = []
    gts_list = []
    for direc in os.listdir(path+'/MOT16/test/'):
        if direc == '.DS_Store':
            continue
        detections_list.append(txt_to_npy(path+'/MOT16/test/' + direc + '/det/det'))

        pic_list = []
        for pic in sorted(os.listdir(path+'/MOT16/test/' + direc + '/img1')):
            pic_list.append(path+'/MOT16/test/' + direc + '/img1/' + pic)
        frames_list.append(pic_list)

        #gts_list.append(txt_to_npy('data/2DMOT2015/train/' + direc + '/gt/gt'))
    return detections_list, frames_list

def gen_mot2017test_npy(path = '/media/data/gauthier'):
    detections_list = []
    frames_list = []
    gts_list = []
    for direc in os.listdir(path+'/MOT17/test/'):
        if direc == '.DS_Store':
            continue
        detections_list.append(txt_to_npy(path+'/MOT17/test/' + direc + '/det/det'))

        pic_list = []
        for pic in sorted(os.listdir(path+'/MOT17/test/' + direc + '/img1')):
            pic_list.append(path+'/MOT17/test/' + direc + '/img1/' + pic)
        frames_list.append(pic_list)

        #gts_list.append(txt_to_npy('data/2DMOT2015/train/' + direc + '/gt/gt'))
    return detections_list, frames_list
