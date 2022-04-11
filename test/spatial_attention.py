import torch.nn.functional as f
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
class Conv1dPaddingSame(nn.Module):
    '''pytorch version of padding=='same'
    ============== ATTENTION ================
    Only work when dilation == 1, groups == 1
    =========================================
    https://github.com/pytorch/pytorch/issues/3867

    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1dPaddingSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.rand((out_channels, 
                                                 in_channels, kernel_size)))
        # nn.Conv1d default set bias=True，so create this param
        self.bias = nn.Parameter(torch.rand(out_channels))
        
    def forward(self, x):
        batch_size, num_channels, length = x.shape
        if length % self.stride == 0:
            out_length = length // self.stride
        else:
            out_length = length // self.stride + 1

        pad = math.ceil((out_length * self.stride + 
                         self.kernel_size - length - self.stride) / 2)
        out = F.conv1d(input=x, 
                       weight = self.weight,
                       stride = self.stride, 
                       bias = self.bias,
                       padding=pad)
        return out


class SpatialAttention(torch.nn.Module):
    def __init__(self,is_train,num_classes=1 ):
        super(SpatialAttention, self).__init__()        
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        self.conv_features = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.conv_features.parameters():
            param.requires_grad = True
        
        self.conv1d = Conv1dPaddingSame(49,1,1,1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.last_layer = nn.Linear(in_features=2*2048,out_features=512)
        self.train_layer_bin = nn.Linear(in_features=512,out_features=1)
        self.train_layer_class = nn.Linear(in_features=2048,out_features=num_classes)
        self.is_train = is_train
        
    def forward(self, old_feature, new_feature):
        
        if self.is_train:
            new_conv = self.conv_features(new_feature)
            old_conv = self.conv_features(old_feature)
        else:
            new_conv = (new_feature)
            old_conv = (old_feature)
        new_conv = new_conv.permute(0,2,3,1).contiguous()
        new_conv = torch.reshape(new_conv,(new_conv.shape[0],49,2048)).contiguous()

        old_conv = old_conv.permute(0,2,3,1).contiguous()
        old_conv = torch.reshape(old_conv,(old_conv.shape[0],49,2048)).contiguous()
        
        
        norm_old_conv = f.normalize(old_conv,dim=-1,p=2)
        norm_new_conv = f.normalize(new_conv,dim=-1,p=2)
        norm_new_conv_t = new_conv.permute(0,2,1).contiguous()
        
        old_match = torch.bmm(norm_old_conv,norm_new_conv_t)
        
        new_match = old_match.permute(0,2,1).contiguous()
        
        new_conv1d = self.conv1d(new_match)
        old_conv1d = self.conv1d(old_match)
            
        
        new_att = torch.reshape(new_conv1d,(new_conv1d.shape[0],49)).contiguous()
        old_att = torch.reshape(old_conv1d,(old_conv1d.shape[0],49)).contiguous()
        
        new_att = self.softmax(new_att)
        old_att = self.softmax(old_att)
        
        new_att = torch.reshape(new_att,(new_att.shape[0],49,1)).contiguous()
        old_att = torch.reshape(old_att,(old_att.shape[0],49,1)).contiguous()

        new_conv_atted = new_conv * new_att
        old_conv_atted = old_conv * old_att

        new_conv_atted_sum = torch.sum(new_conv_atted, dim=1)
        old_conv_atted_sum = torch.sum(old_conv_atted, dim=1)
        
        catted = torch.cat([old_conv_atted_sum,new_conv_atted_sum],dim=1)
        
        out_feature =  F.relu(self.last_layer(catted))
        
        if self.is_train:
            bin_out = self.train_layer_bin(out_feature)
            class_old = self.train_layer_class(old_conv_atted_sum)
            class_new = self.train_layer_class(new_conv_atted_sum)
            return bin_out.view(-1),class_old,class_new
        return out_feature

