
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from core.config import  cfg
import torch
from torch.autograd import Variable


class Affinity_Propagate(nn.Module):
    
    def __init__(self, spn = False):
        super(Affinity_Propagate, self).__init__()
        self.spn = spn
        

    def forward(self, guidance, blur_depth, sparse_depth=None):
        
        # normalize features
        gate1_w1_cmb = guidance.narrow(1,0,1)
        gate2_w1_cmb = guidance.narrow(1,1,1)
        gate3_w1_cmb = guidance.narrow(1,2,1)
        gate4_w1_cmb = guidance.narrow(1,3,1)
        gate5_w1_cmb = guidance.narrow(1,4,1)
        gate6_w1_cmb = guidance.narrow(1,5,1)
        gate7_w1_cmb = guidance.narrow(1,6,1)
        gate8_w1_cmb = guidance.narrow(1,7,1)
        
        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            result_depth = (1- sparse_mask)*blur_depth.clone()+sparse_mask*sparse_depth
        else:
            result_depth = blur_depth.clone()
        
        
        for i in range(cfg.SEM.SPN_ITERS):
        # one propagation
            spn_kernel = 3 
            elewise_max_gate1 = self.eight_way_propagation(gate1_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate2 = self.eight_way_propagation(gate2_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate3 = self.eight_way_propagation(gate3_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate4 = self.eight_way_propagation(gate4_w1_cmb, result_depth, spn_kernel)  
            elewise_max_gate5 = self.eight_way_propagation(gate5_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate6 = self.eight_way_propagation(gate6_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate7 = self.eight_way_propagation(gate7_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate8 = self.eight_way_propagation(gate8_w1_cmb, result_depth, spn_kernel) 
            
            result_depth = self.max_of_8_tensor(elewise_max_gate1, elewise_max_gate2, elewise_max_gate3, elewise_max_gate4,\
                                                elewise_max_gate5, elewise_max_gate6, elewise_max_gate7, elewise_max_gate8)
            if  sparse_depth is not None:
                result_depth = (1- sparse_mask)*result_depth.clone()+sparse_mask*sparse_depth
            else:
                result_depth = result_depth.clone()
    
        return result_depth
       
    
    def eight_way_propagation(self, weight_matrix, blur_matrix, kernel):
        if len(weight_matrix.shape)==5:
            weight_matrix=weight_matrix.squeeze(1)
        b, channels, h, w = blur_matrix.shape
        self.groups=channels
        self.avg_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, groups=self.groups, stride=1, padding=1, bias=False)
        weight = torch.ones(channels, channels//self.groups,  kernel, kernel).cuda()
        weight[:,:,(kernel-1)//2,(kernel-1)//2]=0
        self.avg_conv.weight = nn.Parameter(weight)
        for param in self.avg_conv.parameters():
            param.requires_grad = False

        b, channels, h, w = weight_matrix.shape
        self.groups = channels
        self.sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel,groups=self.groups, stride=1, padding=1, bias=False)
        sum_weight = torch.ones(channels, channels//self.groups, kernel, kernel).cuda()
        self.sum_conv.weight = nn.Parameter(sum_weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False

        weight_abs = torch.abs(weight_matrix)
        weight_sum = self.sum_conv(weight_abs)
        weight_matrix = torch.div(weight_matrix, weight_sum)
        assert torch.sum(weight_matrix>1)*torch.sum(weight_matrix<-1)==0, 'weight_matrix is wrong'
        avg_sum = self.avg_conv(weight_matrix*blur_matrix)
        out = weight_matrix*blur_matrix + avg_sum
        del self.avg_conv
        del self.sum_conv
        del sum_weight
        del weight
        return out
        
    def normalize_gate(self, guidance):
        gate1_x1_g1 = guidance.narrow(1,0,1)
        gate1_x1_g2 = guidance.narrow(1,1,1)
        gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
        gate1_x1_g2_abs = torch.abs(gate1_x1_g2)  
        elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb
    
    
    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)    

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4) 
