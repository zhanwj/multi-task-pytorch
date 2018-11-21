import torch
import torch.nn.functional as F


def EPE(input_flow, target_flow, sparse=False, mean=True, channel=2):
    EPE_map = torch.norm(target_flow-input_flow, channel,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        if channel == 2:
            mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        else:
            mask = target_flow  == 0

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size


def sparse_max_pool(input, size, channel=2):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''
    positive = (input > 0).float()
    if channel == 2:
        negative = (input < 0).float()
        output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    else:
        output = F.adaptive_max_pool2d(input * positive, size)

    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, c, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w), channel=c)
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, sparse, mean=True, channel=c)


    loss = 0
    for output  in network_output:
        loss += one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)
