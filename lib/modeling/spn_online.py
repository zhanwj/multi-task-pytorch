from core.config import  cfg
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_,constant_
from modeling.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

class SPN(nn.Module):
	"""docstring for SPN"""
	def __init__(self):
		super(SPN, self).__init__()
		self.connection_ways=cfg.SEM.SPN_CONNECTION_WAYS
		self.left_to_right=GateRecurrent2dnoind(True,False)
		self.right_to_left=GateRecurrent2dnoind(True,True)
		self.bottom_to_up =GateRecurrent2dnoind(False,True)
		self.up_to_bottom =GateRecurrent2dnoind(False,False)

		self.guide_conv1=nn.Conv2d(cfg.MODEL.NUM_CLASSES,32,kernel_size=3,padding=1,stride=1,bias=False)
		self.guide_conv2=nn.Conv2d(256,384,kernel_size=3,padding=1,stride=1,bias=False)

		self.elt_resize_deconv=nn.Sequential(                      #1/2
		    nn.Conv2d(32,64,3,padding=1,stride=1,bias=False),
		    nn.ReLU(inplace=True),
		    nn.Conv2d(64,cfg.MODEL.NUM_CLASSES,kernel_size=3,padding=1,stride=1,bias=False)
		    )

		for m in self.modules():
		    if isinstance(m,nn.Conv2d):
		        kaiming_normal_(m.weight, 0.01)
		        if m.bias is not None:
		            constant_(m.bias,0)


	def _gNorm(self,glist):
		return_list=[]
		g_norm_list=[]
		sum_abs=glist[0].abs()
		for i in range(1,len(glist)):
			sum_abs+=glist[i].abs()
		mask_need_norm=sum_abs.ge(1)
		mask_need_norm=mask_need_norm.float()

		for i in range(len(glist)):
			g_norm_list.append(torch.div(glist[i],sum_abs))
			return_list.append(torch.add(-mask_need_norm,1)*glist[i]+mask_need_norm*g_norm_list[i])
		return return_list

	def forward(self,featureMap,guidance):
		
		featureMap=self.guide_conv1(featureMap)
		guidance=self.guide_conv2(guidance)
        b,c,w,h = guidance.shape
        guidance=gudiance.view(b,4,3,-1,h,w)
        guidance_sum = torch.sum(torch.abs(gudiance), dim=2).unsqueeze(2)
        mask_need_norm = guidance_sum.ge(1)
        mask_need_norm=mask_need_norm.float()

        guidance_norm = torch.div(guidance, guidance_sum)
        guidance_scan = torch.add(-mask_need_norm, 1)*guidance + mask_need_norm*guidance_norm
		
		G_left_to_right,G_right_to_left,G_bottom_to_up,G_up_to_bottom=torch.split(guidance_scan, split_size_or_sections=1, dim=1)

		left_weight1, left_weigt2, left_weight3 = torch.split(G_left_to_right, split_size_or_sections=1, dim=2).sequeeze(1)
		output_left_to_right=self.left_to_right(featureMap,left_weight1, left_weight2, left_weight3)

		right_weight1, right_weigt2, right_weight3 = torch.split(G_right_to_left, split_size_or_sections=1, dim=2).sequeeze(1)
		output_right_to_left=self.right_to_left(featureMap,right_weight1, right_weight2, right_weight3)

		bottom_weight1, bottom_weigt2, bottom_weight3 = torch.split(G_bottom_to_up, split_size_or_sections=1, dim=2).sequeeze(1)
		output_bottom_to_up =self.bottom_to_up (featureMap,bottom_weight1, bottom_weight2, bottom_weight3)

		up_weight1, up_weigt2, up_weight3 = torch.split(G_up_to_bottom, split_size_or_sections=1, dim=2).sequeeze(1)
		output_up_to_bottom =self.up_to_bottom (featureMap,up_weight1, up_weigt2, up_weight3)

		output_max=torch.max(torch.max(torch.max(output_left_to_right,output_right_to_left),output_up_to_bottom),output_bottom_to_up)
		
		if cfg.SEM.SPN_ITERS>1:
			for i in range(cfg.SEM.SPN_ITERS-1):
                output_left_to_right=self.left_to_right(featureMap,left_weight1, left_weight2, left_weight3)
                output_right_to_left=self.right_to_left(featureMap,right_weight1, right_weight2, right_weight3)
                output_bottom_to_up =self.bottom_to_up (featureMap,bottom_weight1, bottom_weight2, bottom_weight3)
                output_up_to_bottom =self.up_to_bottom (featureMap,up_weight1, up_weigt2, up_weight3)
				output_max=torch.max(torch.max(torch.max(output_left_to_right,output_right_to_left),output_up_to_bottom),output_bottom_to_up)
		#return output_max
		predict1x=self.elt_resize_deconv(output_max)
		return predict1x



	
