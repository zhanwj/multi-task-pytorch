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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO
#cityscapes api
from .import cityscapes_api as cityscapes

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.label_list_root = DATASETS[name][ANN_FN]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.cityscapes = cityscapes
        self.debug_timer = Timer()

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            trainset='train',
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        if cfg.DEBUG:
            roidb = self.cityscapes.load_train_list(self.label_list_root)[:10]
        else:
            roidb = self.cityscapes.load_train_list(self.label_list_root)
        for entry in roidb:
            self._prep_roidb_entry(entry)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path, file_name is the root of: image_L, image_R, semseg, disparity
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name'].split()[0]
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image_L'] = im_path

        # add semseg label name
        file_name = entry['file_name'].split()
        if len(file_name) == 4:
            image_L, image_R, semseg, disp = file_name
        else:
            image_L, image_R = file_name
            semseg, disp = None, None
        im_path = os.path.join(
            self.image_directory, self.image_prefix + image_L
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)

        if semseg is not None:
            im_path = os.path.join(
                self.image_directory, self.image_prefix + semseg
            )
            assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
            entry[cfg.SEM.OUTPUT_PREFIX] = im_path

        if cfg.SPN.SPN_ON:
            im_path = os.path.join(
                cfg.SPN.SAVE_FILE, image_L.split('/')[-1].replace('.png', '_prob.pkl')
            )
            assert os.path.exists(im_path), 'Coarse Image \'{}\' not found'.format(im_path)
            entry['seg_coarse_path'] = im_path

        if cfg.DISP.DISP_ON:
            im_path = os.path.join(
                self.image_directory, self.image_prefix + image_R
            )
            assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
            entry['image_R'] = im_path
            if disp is not None:
                im_path = os.path.join(
                    self.image_directory, self.image_prefix + disp
                )
                assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
                entry[cfg.DISP.OUTPUT_PREFIX] = im_path

        entry['flipped'] = False
        entry['need_crop'] = False
        entry['width']=2048
        entry['height']=1024
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]
