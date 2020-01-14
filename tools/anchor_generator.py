import torch
import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg,  _merge_a_into_b
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from modeling.model_builder import LCRNet
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats
import json
from six.moves import cPickle as pickle
NT=5

def generate_from_model(modelname):
    print(modelname)
    anchor_cfg={}
    with open(modelname, 'rb') as fid:
        model = pickle.load(fid)
    anchor_cfg['anchor_poses'] = model['anchor_poses'][:,:13*NT]
    anchor_cfg['ppi_params']=model['ppi_params']
    anchor_cfg['cfg']=model['cfg']
    print(model.keys())
    with open('configs/anchors/'+modelname.split('/')[-1][:-4]+'_anchor_cfg.pkl', 'wb') as f:
        pickle.dump(anchor_cfg, f)
        print('anchor_cfg.pkl saved ')
        f.close()

if __name__=='__main__':
    name=sys.argv[1]
    generate_from_model('models/'+name+'.pkl')