from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

from torchreid.data import ImageDataset
import random
import copy
class MOT_cropped0(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/0/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped0, self).__init__(train, query, gallery, **kwargs)
        
        
class MOT_cropped1(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/1/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped1, self).__init__(train, query, gallery, **kwargs)
        
        
        
class MOT_cropped2(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/2/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        
        super(MOT_cropped2, self).__init__(train, query, gallery, **kwargs)
        
        
class MOT_cropped3(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/3/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped3, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped4(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/4/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped4, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped5(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/5/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped5, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped6(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/6/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped6, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped7(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/7/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped7, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped8(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/8/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped8, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped9(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/9/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped9, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped10(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/10/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped10, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped11(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/11/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped11, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped12(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/12/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped12, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped13(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/13/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped13, self).__init__(train, query, gallery, **kwargs)

class MOT_cropped14(ImageDataset):
    dataset_dir = '/media/data/gauthier/mot_cropped/14/'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        all_crops = os.listdir(self.dataset_dir)
        tuple_list = []
        
        for crop in all_crops:
            tuple_list += [(self.dataset_dir+crop,int(crop.split('_')[0]),0)]
            
        s_list = sorted(tuple_list, key=lambda x: x[1])
        
        l = len(s_list)
        query_s = copy.deepcopy(s_list[0:int(1*l/10)])
        gallery_s = copy.deepcopy(s_list[0:int(1*l/10)])
        random.shuffle(s_list)
        
        train = s_list
        
        query = query_s
        
        gallery_l = []
        
        for tup in gallery_s:
            gallery_l.append((tup[0],tup[1],tup[2]+1))
        gallery = gallery_l

        super(MOT_cropped14, self).__init__(train, query, gallery, **kwargs)


