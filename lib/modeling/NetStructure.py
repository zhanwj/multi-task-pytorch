
from pspnet_dispSeg import DispSeg
from tensorboardX import SummaryWriter
writer=SummaryWriter(log_dir='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/network_structure/dispSeg/')

class NetStructure(DispSeg):
	def __init__(self):
		super(NetStructure,self).__init__()
	    
	def forward(self,data):
		x=self.pspnet(data)
		pred=self.glassGCN(x)
		return pred
net=NetStructure()
data=torch.ones((4,360,1440)).cuda()
with writer:
	writer.add_graph(net,data)
