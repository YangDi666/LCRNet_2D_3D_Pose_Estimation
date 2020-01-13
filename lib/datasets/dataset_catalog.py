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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    
    
    'real1_train': {
        IM_DIR:
            _DATA_DIR + '/real1/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real1/annotations/instances_train2017.json'
    },
    'real2_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch2/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch2/annotations/instances_train2017.json'
    },
    'real3_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch3/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch3/annotations/instances_train2017.json'
    },
    'real4_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch4/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch4/annotations/instances_train2017.json'
    },
    'real5_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch5/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch5/annotations/instances_train2017.json'
    },
    'real6_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch6/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch6/annotations/instances_train2017.json'
    },
    'real7_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch7/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch7/annotations/instances_train2017.json'
    },
    'real8_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch8/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch8/annotations/instances_train2017.json'
    },
    'real9_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch9/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch9/annotations/instances_train2017.json'
    },
    'real10_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch10/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch10/annotations/instances_train2017.json'
    },
    'real11_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch11/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch11/annotations/instances_train2017.json'
    },
    'real12_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch12/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch12/annotations/instances_train2017.json'
    },
    'real13_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch13/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch13/annotations/instances_train2017.json'
    },
    'real14_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch14/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch14/annotations/instances_train2017.json'
    },
    'real15_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch15/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch15/annotations/instances_train2017.json'
    },
    'real16_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch16/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch16/annotations/instances_train2017.json'
    },
    'real17_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch17/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch17/annotations/instances_train2017.json'
    },
    'real18_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch18/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch18/annotations/instances_train2017.json'
    },
    'real19_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch19/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch19/annotations/instances_train2017.json'
    },
    'real20_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch20/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch20/annotations/instances_train2017.json'
    },
     
    'real21_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch21/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch21/annotations/instances_train2017.json'
    },
    'real22_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch22/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch22/annotations/instances_train2017.json'
    },
    'real23_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch23/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch23/annotations/instances_train2017.json'
    },
    'real24_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch24/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch24/annotations/instances_train2017.json'
    },
    'real25_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch25/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch25/annotations/instances_train2017.json'
    },
    'real26_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch26/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch26/annotations/instances_train2017.json'
    },
    'real27_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch27/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch27/annotations/instances_train2017.json'
    },
    'real28_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch28/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch28/annotations/instances_train2017.json'
    },
    'real29_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch29/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch29/annotations/instances_train2017.json'
    },
    'real30_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch30/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch30/annotations/instances_train2017.json'
    },
    'real31_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch31/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch31/annotations/instances_train2017.json'
    },
    'real32_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch32/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch32/annotations/instances_train2017.json'
    },
    'real33_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch33/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch33/annotations/instances_train2017.json'
    },
    'real34_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch34/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch34/annotations/instances_train2017.json'
    },
    'real35_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch35/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch35/annotations/instances_train2017.json'
    },
    'real36_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch36/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch36/annotations/instances_train2017.json'
    },
      'real37_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch37/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch37/annotations/instances_train2017.json'
    },
    'real38_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch38/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch38/annotations/instances_train2017.json'
    },
    'real39_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch39/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch39/annotations/instances_train2017.json'
    },
    'real40_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch40/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch40/annotations/instances_train2017.json'
    },
    'real41_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch41/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch41/annotations/instances_train2017.json'
    },
    'real42_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch42/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch42/annotations/instances_train2017.json'
    },
    'real43_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch43/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch43/annotations/instances_train2017.json'
    },
    'real44_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch44/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch44/annotations/instances_train2017.json'
    },
    'real45_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch45/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch45/annotations/instances_train2017.json'
    },
    'real46_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch46/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch46/annotations/instances_train2017.json'
    },
    'real47_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch47/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch47/annotations/instances_train2017.json'
    },
    'real48_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch48/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch48/annotations/instances_train2017.json'
    },
    'real49_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch49/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch49/annotations/instances_train2017.json'
    },
    'real50_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch50/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch50/annotations/instances_train2017.json'
    },
    'real51_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch51/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch51/annotations/instances_train2017.json'
    },
    'real52_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch52/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch52/annotations/instances_train2017.json'
    },
    'real53_train': {
        IM_DIR:
            _DATA_DIR + '/real/batch53/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real/batch53/annotations/instances_train2017.json'
    },
    'real54_train': {
        IM_DIR:
            _DATA_DIR + '/real54/images/train2017',
        ANN_FN:
            _DATA_DIR + '/real54/annotations/instances_train2017.json'
    },
    'human3.6m_train': {
        IM_DIR:
            _DATA_DIR + '/human3.6m/images/train2017',
        ANN_FN:
            _DATA_DIR + '/human3.6m/annotations/instances_train2017.json'
    },
    'human3.6m_real_train': {
        IM_DIR:
            _DATA_DIR + '/human3.6m_real/images/train2017',
        ANN_FN:
            _DATA_DIR + '/human3.6m_real/annotations/instances_train2017.json'
    },
    
    'real1_val': {
        IM_DIR:
            _DATA_DIR + '/real1/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real1/annotations/instances_val2017.json'
    },
    'real2_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch2/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch2/annotations/instances_val2017.json'
    },
    'real3_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch3/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch3/annotations/instances_val2017.json'
    },
    'real4_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch4/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch4/annotations/instances_val2017.json'
    },
    'real5_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch5/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch5/annotations/instances_val2017.json'
    },
    'real6_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch6/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch6/annotations/instances_val2017.json'
    },
    'real7_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch7/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch7/annotations/instances_val2017.json'
    },
    'real8_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch8/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch8/annotations/instances_val2017.json'
    },
    'real9_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch9/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch9/annotations/instances_val2017.json'
    },
    'real10_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch10/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch10/annotations/instances_val2017.json'
    },
    'real11_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch11/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch11/annotations/instances_val2017.json'
    },
    'real12_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch12/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch12/annotations/instances_val2017.json'
    },
    'real13_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch13/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch13/annotations/instances_val2017.json'
    },
    'real14_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch14/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch14/annotations/instances_val2017.json'
    },
    'real15_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch15/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch15/annotations/instances_val2017.json'
    },
    'real16_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch16/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch16/annotations/instances_val2017.json'
    },
    'real17_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch17/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch17/annotations/instances_val2017.json'
    },
    'real18_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch18/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch18/annotations/instances_val2017.json'
    },
    'real19_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch19/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch19/annotations/instances_val2017.json'
    },
    'real20_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch20/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch20/annotations/instances_val2017.json'
    },
     
    'real21_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch21/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch21/annotations/instances_val2017.json'
    },
    'real22_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch22/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch22/annotations/instances_val2017.json'
    },
    'real23_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch23/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch23/annotations/instances_val2017.json'
    },
    'real24_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch24/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch24/annotations/instances_val2017.json'
    },
    'real25_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch25/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch25/annotations/instances_val2017.json'
    },
    'real26_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch26/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch26/annotations/instances_val2017.json'
    },
    'real27_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch27/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch27/annotations/instances_val2017.json'
    },
    'real28_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch28/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch28/annotations/instances_val2017.json'
    },
    'real29_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch29/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch29/annotations/instances_val2017.json'
    },
    'real30_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch30/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch30/annotations/instances_val2017.json'
    },
    'real31_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch31/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch31/annotations/instances_val2017.json'
    },
    'real32_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch32/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch32/annotations/instances_val2017.json'
    },
    'real33_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch33/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch33/annotations/instances_val2017.json'
    },
    'real34_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch34/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch34/annotations/instances_val2017.json'
    },
    'real35_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch35/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch35/annotations/instances_val2017.json'
    },
    'real36_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch36/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch36/annotations/instances_val2017.json'
    },
      'real37_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch37/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch37/annotations/instances_val2017.json'
    },
    'real38_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch38/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch38/annotations/instances_val2017.json'
    },
    'real39_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch39/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch39/annotations/instances_val2017.json'
    },
    'real40_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch40/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch40/annotations/instances_val2017.json'
    },
    'real41_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch41/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch41/annotations/instances_val2017.json'
    },
    'real42_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch42/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch42/annotations/instances_val2017.json'
    },
    'real43_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch43/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch43/annotations/instances_val2017.json'
    },
    'real44_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch44/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch44/annotations/instances_val2017.json'
    },
    'real45_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch45/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch45/annotations/instances_val2017.json'
    },
    'real46_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch46/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch46/annotations/instances_val2017.json'
    },
    'real47_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch47/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch47/annotations/instances_val2017.json'
    },
    'real48_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch48/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch48/annotations/instances_val2017.json'
    },
    'real49_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch49/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch49/annotations/instances_val2017.json'
    },
    'real50_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch50/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch50/annotations/instances_val2017.json'
    },
    'real51_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch51/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch51/annotations/instances_val2017.json'
    },
    'real52_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch52/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch52/annotations/instances_val2017.json'
    },
    'real53_val': {
        IM_DIR:
            _DATA_DIR + '/real/batch53/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real/batch53/annotations/instances_val2017.json'
    },
    'real54_val': {
        IM_DIR:
            _DATA_DIR + '/real54/images/val2017',
        ANN_FN:
            _DATA_DIR + '/real54/annotations/instances_val2017.json'
    },
    'human3.6m_val': {
        IM_DIR:
            _DATA_DIR + '/human3.6m/images/val2017',
        ANN_FN:
            _DATA_DIR + '/human3.6m/annotations/instances_val2017.json'
    },
    'human3.6m_real_val': {
        IM_DIR:
            _DATA_DIR + '/human3.6m_real/images/val2017',
        ANN_FN:
            _DATA_DIR + '/human3.6m_real/annotations/instances_val2017.json'
    },
    
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    }
}
