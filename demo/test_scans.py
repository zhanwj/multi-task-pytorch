import torch
maxdisp = 10
disp_scans = torch.arange(maxdisp).view(1,maxdisp,1,1)
zeros_scans = torch.arange(maxdisp).view(1,maxdisp,1,1)
semseg  = torch.arange(16).view(4,4)

zeros_scans = torch.cat([zeros_scans.repeat(1, repeat,1, 1) for i in range(1)],dim=0)
