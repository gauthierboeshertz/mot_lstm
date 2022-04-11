from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import torch
    
from utils import get_bb_content_pil



class dataset_dhn(torch.utils.data.Dataset):
    def __init__(self, data,frames):
        # here is a mapping from this index to the mother ds index
        self.data = data
     #   self.transform = Compose([ToTensor(), Normalize(mean = [0.485, 0.456, 0.406],std= [0.229, 0.224, 0.225])])
        self.frames = frames
       # self.imgs = [Image.open(frame) for frame in frames]
    def __getitem__(self, index):
        return self.data[index], self.frames[index]

    def __len__(self):
        return len(self.data)

    
class simple_dataset(torch.utils.data.Dataset):
    ###data = [(label, input)]
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    
class GenHelper(Dataset):
    def __init__(self, mother, label, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother
        self.label = label
    def __getitem__(self, index):
        return self.mother[self.mapping[index]], self.label[self.mapping[index]]

    def __len__(self):
        return self.length


class GenHelperImage(Dataset):
    def __init__(self, mother, label, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother
        self.label = label
        self.preprocess =transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):

        e = self.mother[self.mapping[index]]
        bb_pil = get_bb_content_pil(e[0][0], e[0][1])
        bb_ret = self.preprocess(bb_pil)
        bb_ret= bb_ret.unsqueeze(0)
        
        for i in range(1,len(e)):
            bb_pil = get_bb_content_pil(e[i][0], e[i][1])
            bb = self.preprocess(bb_pil)
            bb= bb.unsqueeze(0)
            bb_ret = torch.cat((bb_ret,bb))
            
        return bb_ret,self.label[self.mapping[index]]

    def __len__(self):
        return self.length


