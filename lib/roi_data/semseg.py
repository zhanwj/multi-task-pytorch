# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Construct minibatches for Fast R-CNN training. Handles the minibatch blobs
that are specific to Fast R-CNN. Other blobs that are generic to RPN, etc.
are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr

from core.config import cfg
import roi_data.keypoint_rcnn
import roi_data.mask_rcnn
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import cv2
from PIL import Image
def add_sem_blobs(blobs, im_scales, roidb, interp):
    """Add blobs needed for training Semantic segmentation style models."""
    # Sample training semantic label from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i, interp)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)


    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True

    return valid


def _sample_rois(roidb, im_scale, batch_idx, interp):
    """Load a semantic label 
    """
    scale, crop_index = im_scale
    if interp == cv2.INTER_NEAREST:
        prefix = cfg.SEM.OUTPUT_PREFIX
        input_label = 255*np.ones(cfg.SEM.INPUT_SIZE, dtype=np.long)
        semseg_label = cv2.imread(roidb[prefix], 0)
        assert np.any(semseg_label != -1), 'semseg error -1'
    else:
        prefix = cfg.DISP.OUTPUT_PREFIX
        input_label = np.zeros(cfg.SEM.INPUT_SIZE, dtype=np.float32)
        semseg_label = np.asarray(Image.open(roidb[prefix]))/255
    semseg_label = cv2.resize(semseg_label, (scale, scale//2), interpolation=interp)
    if roidb['flipped']:
        semseg_label = semseg_label[:, ::-1]
    y1, x1, h_e, w_e = crop_index
    input_label[0: h_e, 0: w_e] = semseg_label[y1: y1+ h_e, x1: x1+ w_e]
    #input_label = input_label.astype(np.float16)
    blob_dict = {}
    if interp != cv2.INTER_NEAREST:
        blob_dict['{}_{}'.format(prefix, 0)]= \
                input_label[np.newaxis, :, :]
        return blob_dict 
    # Add label to other loss
    blob_dict['{}_{}'.format(prefix, 0)] = input_label[np.newaxis].copy()

    return blob_dict


