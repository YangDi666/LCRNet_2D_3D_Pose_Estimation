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
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import pickle
NT=5

# Load anchor poses
anchorfile='configs/anchors/anchor_cfg.pkl'
with open(anchorfile, 'rb') as fid:
   anchor_cfg = pickle.load(fid)
anchor_poses = anchor_cfg['anchor_poses'][:,:13*NT]

def get_fast_rcnn_blob_names(is_training=True):
    """Fast R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']
    if is_training:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights']
        blob_names += ['bbox_outside_weights']
        # pose_targets blob: R pose regression targets with 65
        # targets per class
        blob_names += ['pose_targets']
        blob_names += ['pose_inside_weights']
        blob_names += ['pose_outside_weights']
       
    
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    return blob_names


def add_fast_rcnn_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        #print(entry.keys())
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    
    return valid


def _sample_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]
    sampled_poses = roidb['poses'][keep_inds]
    
    if 'bbox_targets' not in roidb:
        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        gt_boxes = roidb['boxes'][gt_inds, :]
        gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
        
        bbox_targets = _compute_targets(
            sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels)
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
    else:
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(
            roidb['bbox_targets'][keep_inds, :])

    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)
     
    if 'pose_targets' not in roidb:
        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        gt_poses = roidb['poses'][gt_inds, :]
        
        gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
        
        pose_targets = _compute_pose_targets(
            sampled_poses, gt_poses[gt_assignments, :], sampled_boxes, sampled_labels, anchor_poses)
        pose_targets, pose_inside_weights = _expand_pose_targets(pose_targets)
    else:
        pose_targets, pose_inside_weights = _expand_pose_targets(
            roidb['pose_targets'][keep_inds, :])

    pose_outside_weights = np.array(
        pose_inside_weights > 0, dtype=pose_inside_weights.dtype)


    
        
    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights,
        pose_targets=pose_targets,
        pose_inside_weights=pose_inside_weights,
        pose_outside_weights=pose_outside_weights)

  
    return blob_dict


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    #targets = gt_rois
    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois,
                                           cfg.MODEL.BBOX_REG_WEIGHTS)
    # Use class "1" for all fg boxes if using class_agnostic_bbox_reg
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        labels.clip(max=1, out=labels)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)



def reclass(gt, anchor, labels):
    inds = np.where(labels > 0)[0]
    labels_new=labels
    for i in range(gt[inds,:].shape[0]):
       
        gti=np.tile(gt[i, :], (anchor.shape[0],1))
        dis=(gti-anchor)**2
        dis=np.sqrt(dis.sum(axis=1))
        class_new=dis.argmin(axis=0)+1
        labels_new[i]=class_new
        
    return labels_new
        

def _compute_pose_targets(ex_pose, gt_pose, ex_box, labels, anchor_poses):
    """Compute pose regression targets for an image."""
    assert ex_pose.shape[0] == gt_pose.shape[0]
    assert ex_pose.shape[1] == NT*13
    assert gt_pose.shape[1] == NT*13
    
    boxes_size = ex_box[:,2:4]-ex_box[:,0:2]
    offset = np.concatenate( ( ex_box[:,:2], np.zeros((ex_box.shape[0],3),dtype=np.float32)), axis=1) # x,y top-left corner for each box
    scale = np.concatenate( ( boxes_size[:,:2], np.ones((ex_box.shape[0],3),dtype=np.float32)), axis=1) # width, height for each box
    offset_poses = np.concatenate( [np.tile( offset[:,k:k+1], (1,13)) for k in range(NT)], axis=1) # x,y top-left corner for each pose 
    scale_poses = np.concatenate( [np.tile( scale[:,k:k+1], (1,13)) for k in range(NT)], axis=1)
    # x- y- scale for each pose 
    
    gt_nor=(gt_pose-offset_poses)/scale_poses
    labels=reclass(gt_nor, anchor_poses, labels)
    anchor_poses_class=np.concatenate([[anchor_poses[i-1]] for i in labels], axis=0)
    targets =(gt_pose-offset_poses)/scale_poses - anchor_poses_class # put anchor poses into the boxes
    #targets = gt_rois-ex_rois
    
    # Use class "1" for all fg boxes if using class_agnostic_bbox_reg
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        labels.clip(max=1, out=labels)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)
        
def _get_class_for_person(poses, anchors):
   
    gt_pose=poses.astype(np.float32)
  
    dis=(gt_pose-anchors)**2
    dis_total=np.array([np.sqrt(dis[:,k:k+NT*13].sum(axis=1)) for k in range(anchors.shape[0])])
    class_min=np.argmin(dis_total, axis=0)
    labels=class_min+np.ones(class_min.shape)
    #print('class:', class_min)
    return labels
            
def _compute_pose_targets2(ex_pose, gt_pose, ex_box, labels, anchor_poses):
    assert ex_pose.shape[0] == gt_pose.shape[0]
    assert ex_pose.shape[1] == NT*13
    assert gt_pose.shape[1] == NT*13

    gt_pose=np.tile(gt_pose, (1,anchor_poses.shape[0]))
    boxes_size = ex_box[:,2:4]-ex_box[:,0:2]
    offset = np.concatenate( ( ex_box[:,:2], np.zeros((ex_box.shape[0],3),dtype=np.float32)), axis=1) # x,y top-left corner for each box
    scale = np.concatenate( ( boxes_size[:,:2], np.ones((ex_box.shape[0],3),dtype=np.float32)), axis=1) # width, height for each box
    offset_poses = np.tile( np.concatenate( [np.tile( offset[:,k:k+1], (1,13)) for k in range(NT)], axis=1) , (1,anchor_poses.shape[0]))# x,y top-left corner for each pose 
    scale_poses = np.tile( np.concatenate( [np.tile( scale[:,k:k+1], (1,13)) for k in range(NT)], axis=1), (1,anchor_poses.shape[0]))
    # x- y- scale for each pose 
  
    anchors=np.tile( anchor_poses.reshape(1,-1), (ex_pose.shape[0],1) ) 
    poses_normalized=(gt_pose-offset_poses)/scale_poses
    targets = poses_normalized - anchors # put anchor poses into the boxes
    # Use class "1" for all fg boxes if using class_agnostic_bbox_reg
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        labels.clip(max=1, out=labels)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)



def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights
    

def _expand_pose_targets(pose_target_data):
    
    num_pose_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = pose_target_data[:, 0]
    pose_targets = blob_utils.zeros((clss.size, NT*13 * num_pose_reg_classes))
    pose_inside_weights = blob_utils.zeros(pose_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = NT*13* cls
        end = start + NT*13
        pose_targets[ind, start:end] = pose_target_data[ind, 1:]
        pose_inside_weights[ind, start:end] = (1.0)*NT*13
    return pose_targets, pose_inside_weights

def _expand_pose_targets2(pose_target_data):
    
    num_pose_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  
    clss = pose_target_data[:, 0]
    pose_targets = blob_utils.zeros((clss.size, NT*13 * num_pose_reg_classes))
    pose_inside_weights = blob_utils.zeros(pose_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = NT*13* 1
        pose_targets[ind, 13*NT:] = pose_target_data[ind, 1:]
        end = start + NT*13
        pose_inside_weights[ind, start:] = (1.0)*NT*13*anchor_poses.shape[0]
    return pose_targets, pose_inside_weights


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn_utils.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn_utils.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )

    _distribute_rois_over_fpn_levels('rois')
    