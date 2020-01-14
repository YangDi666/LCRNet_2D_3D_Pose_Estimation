"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch
import numpy as np
import logging
import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg

import utils.logging
from datasets.json_dataset import JsonDataset
import validation
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


def get_pos3d(pos2d, pos3d):
    Fx=1145.04940458804
    Fy=1143.78109572365
    Cx=512.541504956548
    Cy=515.4514869776
    
    chest2dx=((pos2d[10]+pos2d[11])/2+(pos2d[4]+pos2d[5])/2)/2
    chest2dy=((pos2d[23]+pos2d[24])/2+(pos2d[17]+pos2d[18])/2)/2
    d_neck=Fx*(abs(pos3d[10]-pos3d[11])*1000/(abs(pos2d[10]-pos2d[11]))*0.5)
    d_hip=Fx*(abs(pos3d[4]-pos3d[5])*1000/(abs(pos2d[4]-pos2d[5]))*0.5)
    #print('dn, dh: ', d_neck, d_hip)
    
    chest3dz=(d_neck+d_hip)/2
    chest3dx=(chest2dx-Cx)/Fx*chest3dz
    chest3dy=(chest2dy-Cy)/Fy*chest3dz
    #print('chest3d: ', chest3dx, chest3dy, chest3dz)
    p3d=pos3d*1000+np.append(np.append(np.tile(chest3dx,(13)), np.tile(chest3dy,(13))), np.tile(chest3dz,(13)), axis=0)
    #print('pose3d: ', p3d)
    return p3d
    
def human36_p1(pre2d, pre3d, gt2d, gt3d, mode):
    pose3d_pre=get_pos3d(pre2d, pre3d)
    pose3d_gt=get_pos3d(gt2d, gt3d)
    #pose3d_pre=pre3d
    #pose3d_gt=gt3d
    if mode=='align':
        dis=np.sqrt((pose3d_pre[:13]-pose3d_gt[:13])**2+(pose3d_pre[13:26]-pose3d_gt[13:26])**2+(pose3d_pre[26:39]-pose3d_gt[26:39])**2)
    if mode=='abs':
        dis=abs((pose3d_pre[:13]-pose3d_gt[:13]))+abs((pose3d_pre[13:26]-pose3d_gt[13:26]))+abs((pose3d_pre[26:39]-pose3d_gt[26:39]))
    return dis.sum(axis=0)/13
    
if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 21
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset[:4] == "real":
        cfg.TEST.DATASETS = (args.dataset+'_val',)
        cfg.MODEL.NUM_CLASSES = 21
    elif args.dataset == "human3.6m":
        cfg.TEST.DATASETS = (args.dataset+'_val',)
        cfg.MODEL.NUM_CLASSES = 21
    elif args.dataset == "human3.6m_real":
        cfg.TEST.DATASETS = (args.dataset+'_val',)
        cfg.MODEL.NUM_CLASSES = 21
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    #assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True
    dataset_name=cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
    image_ids = dataset.COCO.getImgIds()
    image_ids.sort()
    roidb=dataset.get_roidb(gt=True)
    
    imagedir=dataset.image_directory
    results=validation.get_pose(imagedir, ckpt_path, 0, cfg.MODEL.ANCHOR_POSES)
    #print('results:', results)
    err2d=0
    err3d=0
    th2d=2
    th3d=0.3
    crr2d=0
    crr3d=0
    crr=0
    dis_p1=0

    for n, entry in enumerate(roidb):
       
        pose_gt=entry['poses']
        img_name=entry['image']
        result=results[img_name]
        
        for i in result:
            factor=np.sqrt((pose_gt[0,10]-pose_gt[0,11])**2+(pose_gt[0,23]-pose_gt[0,24])**2)
            dis2d=np.sqrt((i['pose2d'][:13]-pose_gt[0,:13])**2+(i['pose2d'][13:]-pose_gt[0,13:26])**2)/factor
            dis3d=np.sqrt((i['pose3d'][:13]-pose_gt[0,26:39])**2+(i['pose3d'][13:26]-pose_gt[0,39:52])**2+(i['pose3d'][26:39]-pose_gt[0,52:65])**2)
            
            correct2d=np.where(dis2d<=th2d)[0].shape[0]
            correct3d=np.where(dis3d<=th3d)[0].shape[0]
            crr2d+=correct2d
            crr3d+=correct3d
            crr+=13
            err2d+=dis2d.sum(axis=0)
            err3d+=dis3d.sum(axis=0)
            dis_p1+=human36_p1(i['pose2d'], i['pose3d'], pose_gt[0,:26], pose_gt[0,26:], mode='align')
    
    print('Err2d : ', err2d)
    print('Err3d : ', err3d*1000)
    print('Err2d_avg : ', err2d/len(roidb))
    print('Err3d_avg : ', err3d/len(roidb))
    print('PCK2d : ', crr2d/crr)
    print('PCK3d : ', crr3d/crr)
    #print('Err3d_human3.6_P1 : ', dis_p1/len(roidb))    
        
    # Save file
    eva_dir = os.path.join('evaluations')
    if not os.path.exists(eva_dir):
        os.makedirs(eva_dir)
    file_handle = open(eva_dir+'/evaluation_'+ckpt_path.split('/')[-1][:-4]+'_'+args.dataset+'.txt', mode='w')
    file_handle.writelines(['Err2d : ', str(err2d), '\n', 'Err3d : ', str(err3d*1000), '\n', 'Err2d_avg : ', str(err2d/len(roidb)), '\n', 'Err3d_avg : ', str(err3d/len(roidb)), '\n', 'PCK2d(2.0) : ', str(crr2d/crr), '\n', 'PCK3d(0.3m) : ', str(crr3d/crr), '\n'])
    file_handle.close()