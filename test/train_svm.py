import numpy as np
from sklearn.svm import SVC
from utils import overlap
from PIL import Image
from libsvm.python.svm import *
from libsvm.python.svmutil import *


def train_svm_active(det, gt, img, threshold_neg, threshold_pos,max_width =0, max_height=0, max_score=0, svm_type="libsvm"):    

    det, gt, labels = gen_data_for_active_svm(det, gt, threshold_neg, threshold_pos)
    
    img_height = img.size[1]
    img_width = img.size[0]
    
    det = det[labels != 0]
    labels = labels[labels != 0]
    max_width =np.max(det[:, 4])
    max_height =np.max(det[:, 5])
    max_score = np.max(det[:, 6])

    norm_det = normalize_active_features(det = det,width= img_width, height= img_height,
                                         max_width=max_width, max_height = max_height, max_score =max_score)
    ones_add = np.ones((norm_det.shape[0],1))
    
    norm_det = np.hstack((norm_det,ones_add))
    
    if svm_type == 'svc':
        svm = SVC(kernel='linear', C=1)
        svm.fit(norm_det, labels)
        
    if svm_type == 'libsvm':
        lnormdet = []
        for ndet in norm_det :
            lnormdet.append(ndet.tolist())

        prob  = svm_problem(labels ,lnormdet)
        param = svm_parameter( '-c 1 -q')
        svm = svm_train(prob, param)
        
    return svm,max_height,max_width,max_score

def normalize_active_features(det, width, height, max_width, max_height, max_score):
    norm_det = np.zeros_like(det)
    for i in range(det.shape[0]):
        norm_det[i][2] = det[i][2] / width
        norm_det[i][3] = det[i][3] / height
        norm_det[i][4] = det[i][4] / max_width
        norm_det[i][5] = det[i][5] / max_height
        norm_det[i][6] = det[i][6] / max_score
    return norm_det[:,2:7]


def gen_data_for_active_svm(det, gt, threshold_neg, threshold_pos):
    labels = np.zeros(det.shape[0])

    for i,detection in enumerate(det):
        frame = detection[0]
        gt_for_frame = gt[gt[:, 0] == frame]
        if gt_for_frame.shape[0] == 0:
            labels[i] = -1
        else:
            o,_,_ = overlap(detection, gt_for_frame)

            max_metric = np.max(o)
            if max_metric < threshold_neg:
                labels[i] = -1
            elif max_metric > threshold_pos:
                labels[i] = 1
            else:
                labels[i] = 0
    return det, gt, labels


