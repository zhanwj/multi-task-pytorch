from roi_data.loader import RoiDataLoader, MinibatchSampler
import torch
import numpy as np
ratio_list=np.array([1,2,3,4,5])
ratio_index=np.array([0,1,2,3,4])
sampler = MinibatchSampler(ratio_list, ratio_index)
batch_sampler = iter(torch.utils.data.sampler.BatchSampler(sampler, batch_size=2, drop_last=False))
for i in range(10):
    items=[]
    for ids, _ in next(batch_sampler):
        items.append(ids)

    print (i, items)
