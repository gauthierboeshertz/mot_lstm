from collections import namedtuple, defaultdict
import hashlib
import numpy as np

Box = namedtuple('Box', ['video_name', 'frame_number', 'obj_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])

def calculate_hash(obj):
    return hashlib.sha256(obj.__repr__().encode("utf-8")).hexdigest()

def box_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = [boxA.bb_left, boxA.bb_top, boxA.bb_width, boxA.bb_height]
    boxB = [boxB.bb_left, boxB.bb_top, boxB.bb_width, boxB.bb_height]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[0] + boxA[2] - boxA[0] + 1) * (boxA[1]+boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[0] + boxB[2] - boxB[0] + 1) * (boxB[1]+boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[0] + boxA[2] - boxA[0] + 1) * (boxA[1]+boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[0] + boxB[2] - boxB[0] + 1) * (boxB[1]+boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    assert iou >= 0
    assert iou <= 1.0
    return iou


def bb_intersection_over_area(boxA, boxB):
    # retuns the percentage of the boxA being occluded by boxB

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[0] + boxA[2] - boxA[0] + 1) * (boxA[1]+boxA[3] - boxA[1] + 1)

    ioa = interArea / float(boxAArea)

    # return the intersection over area value
    assert ioa >= 0
    assert ioa <= 1.0
    return ioa



def box_intersection_over_area(boxA, boxB):
    # retuns the percentage of the boxA being occluded by boxB
    boxA = [boxA.bb_left, boxA.bb_top, boxA.bb_width, boxA.bb_height]
    boxB = [boxB.bb_left, boxB.bb_top, boxB.bb_width, boxB.bb_height]

    return bb_intersection_over_area(boxA, boxB)

def crop_bbox_to_frame(bbox, frame_h, frame_w):
    x, y, w, h = bbox


    if x < 0:
        w = w - abs(x)  # reduce width for the amount outside of border
        x = 0

    if y < 0:
        w = h - abs(y)  # reduce height for the amount outside of border
        y = 0

    w = min(w, frame_w-x)
    h = min(h, frame_h-y)

    assert w > 0
    assert h > 0
    assert x >= 0
    assert y >= 0
    assert x + w <= frame_w
    assert y + h <= frame_h
    bbox = np.array([x,y,w,h])

    return bbox


def box_to_bbox(box):
    return np.array([box.bb_left, box.bb_top, box.bb_width, box.bb_height])

def track_to_box(track):
    box = Box('e', * track[0:7])
    return box

def box_center(box):
    return np.array([box.bb_left+box.bb_width/2, box.bb_top+box.bb_height/2])


def bbox_center(bbox):
    return np.hstack([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])


def calc_centers(bboxes):
    return np.vstack([bboxes[:, 0] + bboxes[:, 2] / 2, bboxes[:, 1] + bboxes[:, 3] / 2]).T


def parse_boxes(in_boxes, video_name, frame_h, frame_w):
    boxes = list()
    for box in in_boxes:
        x, y, w, h = box[2:6]

        if x+w < 0 or x > frame_w or y+h < 0 or y > frame_h:
            continue

        if x < 0:
            w = w - abs(x)  # reduce width for the amount outside of border
            x = 0

        if y < 0:
            h = h - abs(y)  # reduce height for the amount outside of border
            y = 0

        w = min(w, frame_w-x)
        h = min(h, frame_h-y)

        assert w > 0
        assert h > 0
        assert x >= 0
        assert y >= 0
        assert x + w <= frame_w
        assert y + h <= frame_h

        box[2:6] = x, y, w, h
        new_b = [-1 for i in range(7)]
        new_b[0] = float(box[0] )
        new_b[1] = float(box[1] )
        new_b[2:6] =  x, y, w, h
        new_b[6] =  11
        box = Box(video_name, *new_b)
        boxes.append(box)

    return boxes



def create_sequence_gt_boxes_dict(gt_boxes):
    gt_boxes_dict = defaultdict(defaultdict)

    for gt_box in gt_boxes:
        assert type(gt_box) == Box
        gt_boxes_dict[gt_box.frame_number][gt_box.obj_id]=gt_box

    gt_boxes_dict = dict(gt_boxes_dict)
    return gt_boxes_dict


