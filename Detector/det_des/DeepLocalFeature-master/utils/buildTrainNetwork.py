import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import time
import cv2
from tqdm import tqdm
import pickle 

from det_tools import *
from common.tf_layer_utils import *
from common.tf_train_utils import get_optimizer, get_piecewise_lr, get_activation_fn
from common.tfvisualizer import log_images, convert_tile_image

from inference import *

MODEL_PATH = '../models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

def apply_scale_on_intrinsic(K, sx, sy):
    # K : [B,3,3]
    batch_size = tf.shape(K)[0]
    # 3x3 matrix
    S = tf.stack(
        [sx, 0, 0,
         0, sy, 0,
         0, 0, 1])
    S = tf.cast(tf.reshape(S, [3,3]), tf.float32)
    S = tf.tile(S[None], [batch_size, 1, 1])
    return tf.matmul(S, K)

def build_training_network(config, next_batch, is_training, psf, global_step):

    max_outputs = 5
    axis123 = list(range(1,4)) # 1,2,3
    photos1, photos2, depths1, depths2, valid_masks1, valid_masks2, c2Tc1s, c1Tc2s, c1Tws, c2Tws, Ks1, Ks2, thetas1, thetas2, inv_thetas1, inv_thetas2, theta_params = next_batch
    raw_photos1 = tf.identity(photos1)
    raw_photos2 = tf.identity(photos2)
    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photos1)
        photos2 = instance_normalization(photos2)

    batch_size = tf.shape(photos1)[0]
    crop_radius = config.crop_radius
    patch_size = config.patch_size
    mining_type = config.mining_type.lower()
    det_loss_type = config.det_loss.lower()
    desc_loss_type = config.desc_loss.lower()
    K = config.top_k

    # Show tensorboard
    c_red = tf.constant([1,0,0], dtype=tf.float32)
    c_green = tf.constant([0,1,0], dtype=tf.float32)
    c_blue = tf.constant([0,0,1], dtype=tf.float32)
    rgbs1 = tf.concat([raw_photos1, raw_photos1, raw_photos1], axis=-1)
    rgbs2 = tf.concat([raw_photos2, raw_photos2, raw_photos2], axis=-1)

    #----------------------------------
    #  Detector
    #----------------------------------
    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.use_nms3d:
        print('Apply 3D NMS instead.')
        heatmaps1, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photos1, reuse=False)
        heatmaps2, det_endpoints2 = build_multi_scale_deep_detector_3DNMS(config, detector, photos2, reuse=True)
    else:
        heatmaps1, det_endpoints = build_multi_scale_deep_detector(config, detector, photos1, reuse=False)
        heatmaps2, det_endpoints2 = build_multi_scale_deep_detector(config, detector, photos2, reuse=True)

    for i, score_maps in enumerate(det_endpoints['score_maps_list']):
        tf.summary.image('logits1_{}'.format(i), score_maps, max_outputs=max_outputs)
    tf.summary.histogram('heatmaps1', heatmaps1)

    #------------------------------------------------------
    #  Score loss (warp heatmaps and take loss)
    #------------------------------------------------------

    # Heatmap transfer one another
    heatmaps1w, visible_masks1, xy_maps1to2 = \
        inverse_warp_view_2_to_1(heatmaps2, depths2, depths1, c2Tc1s, 
                                K1=Ks1, K2=Ks2, 
                                inv_thetas1=inv_thetas1, thetas2=thetas2,
                                depth_thresh=config.depth_thresh)
    heatmaps2w, visible_masks2, xy_maps2to1 = \
        inverse_warp_view_2_to_1(heatmaps1, depths1, depths2, c1Tc2s, 
                                K1=Ks2, K2=Ks1,
                                inv_thetas1=inv_thetas2, thetas2=thetas1,
                                depth_thresh=config.depth_thresh)
    visible_masks1 = visible_masks1 * valid_masks1 # take 'and'
    visible_masks2 = visible_masks2 * valid_masks2
    
    heatmaps1w.set_shape(heatmaps2.get_shape().as_list())
    heatmaps1w = tf.stop_gradient(heatmaps1w) # to be safe
    heatmaps2w.set_shape(heatmaps1.get_shape().as_list())
    heatmaps2w = tf.stop_gradient(heatmaps2w) # to be safe

    nms_maps1w = non_max_suppression(heatmaps1w, config.nms_thresh, config.nms_ksize)
    nms_maps2w = non_max_suppression(heatmaps2w, config.nms_thresh, config.nms_ksize)
    nms_score1w = heatmaps1w * nms_maps1w # not filter out with mask because this tensor are used to compare with heatmaps
    nms_score2w = heatmaps2w * nms_maps2w
    top_k1w = make_top_k_sparse_tensor(nms_score1w, k=K)
    top_k1w = top_k1w * nms_maps1w
    top_k1w = tf.stop_gradient(top_k1w)
    top_k2w = make_top_k_sparse_tensor(nms_score2w, k=K)
    top_k2w = top_k2w * nms_maps2w
    top_k2w = tf.stop_gradient(top_k2w)

    topk1_canvas = (1.0-det_endpoints['top_ks']) * rgbs1 + det_endpoints['top_ks'] * c_red
    topk2_canvas = (1.0-det_endpoints2['top_ks']) * rgbs2 + det_endpoints2['top_ks'] * c_green
    tf.summary.image('TOPK1-TOPK2', tf.concat([topk1_canvas, topk2_canvas], axis=2), max_outputs=max_outputs)

    tgt_heatmaps1 = heatmaps1
    tgt_heatmaps2 = heatmaps2

    ## regenerate GT-heatmaps otherwise DET outputs goes blur
    gt_heatmaps1 = tf.nn.conv2d(top_k1w, psf, [1,1,1,1], padding='SAME')
    gt_heatmaps1 = tf.minimum(gt_heatmaps1, 1.0)
    gt_heatmaps2 = tf.nn.conv2d(top_k2w, psf, [1,1,1,1], padding='SAME')
    gt_heatmaps2 = tf.minimum(gt_heatmaps2, 1.0)

    Nvis1 = tf.maximum(tf.reduce_sum(visible_masks1, axis=axis123), 1.0)
    Nvis2 = tf.maximum(tf.reduce_sum(visible_masks2, axis=axis123), 1.0)

    if det_loss_type == 'l2loss':
        l2diff1 = tf.squared_difference(tgt_heatmaps1, gt_heatmaps1)
        loss1 = tf.reduce_mean( tf.reduce_sum(l2diff1 * visible_masks1, axis=axis123) / Nvis1 ) 

        l2diff2 = tf.squared_difference(tgt_heatmaps2, gt_heatmaps2)
        loss2 = tf.reduce_mean( tf.reduce_sum(l2diff2 * visible_masks2, axis=axis123) / Nvis2 ) 

        det_loss = (loss1 + loss2) / 2.0
    else:
        raise ValueError('Unknown det_loss: {}'.format(det_loss_type))

    tf.summary.scalar('score_loss', det_loss)

    #------------------------------------------------------
    #  Orientation loss (warp orientation and take loss)
    #------------------------------------------------------
    aug_ori2 = theta_params[:, 3] if config.rot_aug else 0 # rot-aug are applied only on image2
    intheta_c2Rc1 = theta_params[:, 4]
    dori_1to2 = (intheta_c2Rc1 + aug_ori2)[:,None,None,None]

    ori_maps1 = det_endpoints['ori_maps']
    ori_maps2 = det_endpoints2['ori_maps']

    degree_maps1, atan_maps1 = get_degree_maps(ori_maps1)
    degree_maps2, atan_maps2 = get_degree_maps(ori_maps2)
    atan_maps2w = nearest_neighbor_sampling(atan_maps1+dori_1to2, xy_maps2to1) # warp from 1 to 2
    atan_maps1w = nearest_neighbor_sampling(atan_maps2-dori_1to2, xy_maps1to2) # warp from 2 to 1

    ori_maps2w = tf.concat([tf.cos(atan_maps2w), tf.sin(atan_maps2w)], axis=-1)
    ori_maps1w = tf.concat([tf.cos(atan_maps1w), tf.sin(atan_maps1w)], axis=-1)

    angle2rgb = tf.constant(get_angle_colorbar())
    degree_diff1 = tf.reduce_sum(ori_maps1 * ori_maps1w, axis=-1, keep_dims=True)
    degree_diff1 = tf.acos(degree_diff1) # radian
    degree_diff1 = tf.cast(tf.clip_by_value(degree_diff1*180/np.pi+180, 0, 360), tf.int32) 
    degree_diff1 = tf.gather(angle2rgb, degree_diff1[...,0])

    degree_diff2 = tf.reduce_sum(ori_maps2 * ori_maps2w, axis=-1, keep_dims=True)
    degree_diff2 = tf.acos(degree_diff2) # radian
    degree_diff2 = tf.cast(tf.clip_by_value(degree_diff2*180/np.pi+180, 0, 360), tf.int32) 
    degree_diff2 = tf.gather(angle2rgb, degree_diff2[...,0])

    degree_maps1w, _ = get_degree_maps(ori_maps1w)
    degree_maps2w, _ = get_degree_maps(ori_maps2w)

    degree_canvas = tf.concat([
            tf.concat([degree_maps1, degree_maps1w, degree_diff1], axis=2),
            tf.concat([degree_maps2, degree_maps2w, degree_diff2], axis=2),
        ], axis=1)
    tf.summary.image('degree_maps', degree_canvas, max_outputs=max_outputs)

    if config.ori_loss == 'l2loss':
        ori_loss1 = tf.squared_difference(ori_maps1, ori_maps1w)
        ori_loss1 = tf.reduce_mean( tf.reduce_sum(ori_loss1 * visible_masks1, axis=axis123) / Nvis1 ) 
        ori_loss2 = tf.squared_difference(ori_maps2, ori_maps2w)
        ori_loss2 = tf.reduce_mean( tf.reduce_sum(ori_loss2 * visible_masks2, axis=axis123) / Nvis2 ) 
        ori_loss = (ori_loss1 + ori_loss2) * 0.5
    elif config.ori_loss == 'cosine':
        ori_loss1 = tf.reduce_sum(ori_maps1 * ori_maps1w, axis=-1, keep_dims=True) # both ori_maps have already normalized
        ori_loss1 = tf.reduce_mean( tf.reduce_sum(tf.square(1.0-ori_loss1) * visible_masks1, axis=axis123) / Nvis1)
        ori_loss2 = tf.reduce_mean(ori_maps2 * ori_maps2w, axis=-1, keep_dims=True)
        ori_loss2 = tf.reduce_mean( tf.reduce_sum(tf.square(1.0-ori_loss2) * visible_masks2, axis=axis123) / Nvis2)
        ori_loss = (ori_loss1 + ori_loss2) * 0.5
    else:
        raise ValueError('Unknown ori_loss: {}'.format(config.ori_loss))

    tf.summary.scalar('ori_loss_{}'.format(config.ori_loss), ori_loss)

    #------------------------------------------------------
    #  Scale loss (warp orientation and take loss)
    #------------------------------------------------------
    fx1 = tf.reshape(tf.slice(Ks1, [0,0,0], [-1,1,1]), [-1]) # assume fx == fy
    fx2 = tf.reshape(tf.slice(Ks2, [0,0,0], [-1,1,1]), [-1])
    ones = tf.ones_like(depths1)
    aug_scale2 = tf.exp(theta_params[:,1]) if config.scale_aug else 1.0
    scale_maps1 = det_endpoints['scale_maps'][...,None] # [B,H,W,1]
    scale_maps2 = det_endpoints2['scale_maps'][...,None]
    depths1w = nearest_neighbor_sampling(depths2, xy_maps1to2)
    depths1w = tf.where(tf.greater(depths1w, 500), ones, depths1w) # invalid depths are suppressed by 1
    depths2w = nearest_neighbor_sampling(depths1, xy_maps2to1)
    depths2w = tf.where(tf.greater(depths2w, 500), ones, depths2w) 
    scale_maps2w = scale_maps1 * tf.reshape(fx2/fx1*aug_scale2, [-1,1,1,1]) * depths1 / (depths1w+1e-6)
    scale_maps2w = nearest_neighbor_sampling(scale_maps2w, xy_maps2to1)
    scale_maps2w = tf.clip_by_value(scale_maps2w, config.net_min_scale, config.net_max_scale)
    scale_maps2w = tf.stop_gradient(scale_maps2w)
    scale_maps1w = scale_maps2 * tf.reshape(fx1/fx2/aug_scale2, [-1,1,1,1]) * depths2 / (depths2w+1e-6)
    scale_maps1w = nearest_neighbor_sampling(scale_maps1w, xy_maps1to2)
    scale_maps1w = tf.clip_by_value(scale_maps1w, config.net_min_scale, config.net_max_scale)
    scale_maps1w = tf.stop_gradient(scale_maps1w)

    # logscale L2 loss
    scale_loss1 = tf.squared_difference(tf.log(scale_maps1), tf.log(scale_maps1w))
    max_scale_loss1 = tf.reduce_max(scale_loss1)
    scale_loss1 = tf.reduce_mean(tf.reduce_sum(scale_loss1 * visible_masks1, axis=axis123) / Nvis1)
    scale_loss2 = tf.squared_difference(tf.log(scale_maps2), tf.log(scale_maps2w))
    max_scale_loss2 = tf.reduce_max(scale_loss2)
    scale_loss2 = tf.reduce_mean(tf.reduce_sum(scale_loss2 * visible_masks2, axis=axis123) / Nvis2)
    scale_loss = (scale_loss1 + scale_loss2) * 0.5
    tf.summary.scalar('scale_loss', scale_loss)
    det_endpoints['scale_loss'] = scale_loss

    scale_canvas = tf.concat([det_endpoints['scale_maps'], det_endpoints2['scale_maps']], axis=2)[...,None]
    tf.summary.image('Scalemaps1-2', scale_canvas, max_outputs=max_outputs)

    #----------------------------------
    #  Extract patches
    #----------------------------------
    kpts1 = det_endpoints['kpts']
    kpts2 = det_endpoints2['kpts']
    kpts1_int = tf.cast(kpts1, tf.int32)
    kpts2_int = tf.cast(kpts2, tf.int32)
    kpts_scale1 = det_endpoints['kpts_scale']
    kpts_scale2 = det_endpoints2['kpts_scale']
    kpts_ori1 = det_endpoints['kpts_ori']
    kpts_ori2 = det_endpoints2['kpts_ori']

    num_kpts1 = det_endpoints['num_kpts']    
    batch_inds1 = det_endpoints['batch_inds']

    kpts2w = batch_gather_keypoints(xy_maps1to2, batch_inds1, kpts1_int)
    kpts2w_int = tf.cast(kpts2w, tf.int32)
    kpvis2w = batch_gather_keypoints(visible_masks1, batch_inds1, kpts1_int)[:,0] # or visible_masks2, batch_inds2, kpts2w


    kpts_scale2w = batch_gather_keypoints(det_endpoints2['scale_maps'], batch_inds1, kpts2w_int)
    kpts_ori2w = batch_gather_keypoints(ori_maps2, batch_inds1, kpts2w_int)

    # visuaplization of orientation
    cos_maps1 = tf.slice(ori_maps1, [0,0,0,0], [-1,-1,-1,1])
    sin_maps1 = tf.slice(ori_maps1, [0,0,0,1], [-1,-1,-1,1])
    atan_maps1 = tf.atan2(sin_maps1, cos_maps1)
    cos_maps2 = tf.slice(ori_maps2, [0,0,0,0], [-1,-1,-1,1])
    sin_maps2 = tf.slice(ori_maps2, [0,0,0,1], [-1,-1,-1,1])
    atan_maps2 = tf.atan2(sin_maps2, cos_maps2)
    angle2rgb = tf.constant(get_angle_colorbar())
    degree_maps1 = tf.cast(tf.clip_by_value(atan_maps1*180/np.pi+180, 0, 360), tf.int32) 
    degree_maps1 = tf.gather(angle2rgb, degree_maps1[...,0])
    degree_maps2 = tf.cast(tf.clip_by_value(atan_maps2*180/np.pi+180, 0, 360), tf.int32) 
    degree_maps2 = tf.gather(angle2rgb, degree_maps2[...,0])
    degree_maps = tf.concat([degree_maps1, degree_maps2], axis=2)
    tf.summary.image('ori_maps_degree', degree_maps, max_outputs=max_outputs)

    # extract patches
    kp_patches1 = build_patch_extraction(config, det_endpoints, photos1)
    kp_patches2 = build_patch_extraction(config, det_endpoints2, photos2)

    det_endpoints2w = {
        'batch_inds': batch_inds1,
        'kpts': kpts2w,
        'kpts_scale': kpts_scale2w,
        'kpts_ori': kpts_ori2w,
        'feat_maps': det_endpoints2['feat_maps'],
    }
    kp_patches1_pos = build_patch_extraction(config, det_endpoints2w, photos2) # positive pair of kp1

    # Add supervision for orientation
    kpts_ori2w_gt = batch_gather_keypoints(ori_maps2w, batch_inds1, kpts2w_int)
    
    # Visualize patches
    det_endpoints2w_gt = {
        'batch_inds': batch_inds1,
        'kpts': kpts2w,
        'kpts_scale': kpts_scale2w,
        'kpts_ori': kpts_ori2w_gt,
        'feat_maps': det_endpoints2['feat_maps'],
    }
    kp_patches1_pos_gt = build_patch_extraction(config, det_endpoints2w_gt, photos2) # positive pair of kp1

    patches1_canvas = tf.reduce_max(kp_patches1, axis=-1, keep_dims=True) # need channel compression in case feat_maps are not photos
    patches1_pos_canvas = tf.reduce_max(kp_patches1_pos, axis=-1, keep_dims=True)
    patches1_pos_gt_canvas = tf.reduce_max(kp_patches1_pos_gt, axis=-1, keep_dims=True)
    app_patches = tf.concat([patches1_canvas, patches1_pos_canvas * kpvis2w[:,None,None,None], patches1_pos_gt_canvas * kpvis2w[:,None,None,None]], axis=2) # anchor, positive, negative
    app_patches = tf.random_shuffle(app_patches)
    app_patches = convert_tile_image(app_patches[:64])
    app_patches = tf.clip_by_value(app_patches, 0, 1)
    tf.summary.image('GT_app_patches', app_patches, max_outputs=1)

    #----------------------------------
    #  Descriptor
    #----------------------------------
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)

    desc_feats1, desc_endpoints = build_deep_descriptor(config, descriptor, kp_patches1, reuse=False) # [B*K,D]
    desc_feats2, _              = build_deep_descriptor(config, descriptor, kp_patches2, reuse=True)
    desc_feats1_pos, _             = build_deep_descriptor(config, descriptor, kp_patches1_pos, reuse=True)

    tf.summary.histogram('desc_feats1', desc_feats1)

    ## Negative samples selection
    if mining_type == 'hard':
        _, neg_inds = find_hard_negative_from_myself_less_memory(desc_feats1, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1, neg_inds)
    elif mining_type == 'random':
        neg_inds = find_random_negative_from_myself_less_memory(desc_feats1, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1, neg_inds)
    elif mining_type == 'hard2':
        print('Mine hardest negative sample from image2')
        print('[WARNING] find_hard_negative_from_myself_less_memory has bug. it try to search the closest samples from feat2 but it should search from feat1')
        _, neg_inds = find_hard_negative_from_myself_less_memory(desc_feats1_pos, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'hard2geom':
        # too difficult to train because negative is more similar to anchor than positive 
        # geom_sq_thresh = config.hard_geom_thresh ** 2
        # print('Mine hardest negative sample from image2 and geometric constrain (thresh={}, square={})'.format(config.hard_geom_thresh, geom_sq_thresh))
        # _, neg_inds = find_hard_negative_from_myself_with_geom_constrain_less_memory(
        #                 desc_feats1, desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        # desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        # kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
        geom_sq_thresh = config.hard_geom_thresh ** 2
        print('Mine hardest negative sample from image2 and geometric constrain (thresh={}, square={})'.format(config.hard_geom_thresh, geom_sq_thresh))
        _, neg_inds = imperfect_find_hard_negative_from_myself_with_geom_constrain_less_memory(
                        desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'random2':
        print('Mine random negative sample from image2')
        neg_inds = find_random_negative_from_myself_less_memory(desc_feats1_pos, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'rand_hard':
        num_pickup = config.init_num_mine # e.g. 512 // 10
        print('Random Hard Mining #pickup={}'.format(num_pickup))
        geom_sq_thresh = config.hard_geom_thresh ** 2
        neg_inds = find_random_hard_negative_from_myself_with_geom_constrain_less_memory(
                        num_pickup, desc_feats1, desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'rand_hard_sch':
        print('Random Hard Mining with scheduling #pickup={}-->{} (decay={})'.format(config.init_num_mine, config.min_num_pickup, config.pickup_delay))
        num_pickup = tf.maximum(tf.cast(tf.train.exponential_decay(float(config.init_num_mine), global_step, 1000, config.pickup_delay), tf.int32), config.min_num_pickup) # stop decay @ num_pickup=1
        tf.summary.scalar('num_negative_mining', num_pickup)
        geom_sq_thresh = config.hard_geom_thresh ** 2
        neg_inds = find_random_hard_negative_from_myself_with_geom_constrain_less_memory(
                        num_pickup, desc_feats1, desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    else:
        raise ValueError('Unknown mining_type: {}'.format(mining_type))

    if desc_loss_type == 'triplet':
        desc_margin = config.desc_margin
        d_pos = tf.reduce_sum(tf.square(desc_feats1-desc_feats1_pos), axis=1) # [B*K,]
        d_neg = tf.reduce_sum(tf.square(desc_feats1-desc_feats1_neg), axis=1) # [B*K,]

        d_pos = kpvis2w * d_pos # ignore unvisible anchor-positve pairs

        desc_loss = tf.reduce_mean(tf.maximum(0., desc_margin+d_pos-d_neg))
        desc_pair_loss = tf.reduce_mean(d_pos)
        desc_dist_pos = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
        desc_dist_neg = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
        tf.summary.scalar('desc_triplet_loss', desc_loss)
        tf.summary.scalar('desc_pair_loss', desc_pair_loss)
        tf.summary.scalar('dist_pos', desc_dist_pos)
        tf.summary.scalar('dist_neg', desc_dist_neg)
    else:
        raise ValueError('Unknown desc_loss: {}'.format(desc_loss_type))

    patches1_canvas = tf.reduce_max(kp_patches1, axis=-1, keep_dims=True) # need channel compression in case feat_maps are not photos
    patches1_pos_canvas = tf.reduce_max(kp_patches1_pos, axis=-1, keep_dims=True)
    patches1_neg_canvas = tf.reduce_max(kp_patches1_neg, axis=-1, keep_dims=True)
    apn_patches = tf.concat([patches1_canvas, patches1_pos_canvas * kpvis2w[:,None,None,None], patches1_neg_canvas], axis=2) # anchor, positive, negative
    apn_patches = tf.random_shuffle(apn_patches)
    apn_patches = convert_tile_image(apn_patches[:64])
    apn_patches = tf.clip_by_value(apn_patches, 0, 1)
    tf.summary.image('apn_patches', apn_patches, max_outputs=1)

    desc_endpoints['loss'] = desc_loss
    desc_endpoints['feats1'] = desc_feats1
    desc_endpoints['feats2'] = desc_feats2
    desc_endpoints['dist_pos'] = desc_dist_pos
    desc_endpoints['dist_neg'] = desc_dist_neg
    desc_endpoints['kpts1'] = kpts1
    desc_endpoints['kpts2'] = kpts2
    desc_endpoints['kpts2w'] = kpts2w
    desc_endpoints['kpts_scale1'] = kpts_scale1
    desc_endpoints['kpts_scale2'] = kpts_scale2
    desc_endpoints['kpts_scale2w'] = kpts_scale2w
    desc_endpoints['kpts_ori1'] = kpts_ori1
    desc_endpoints['kpts_ori2'] = kpts_ori2
    desc_endpoints['kpts_ori2w'] = kpts_ori2w
    desc_endpoints['kpvis2w'] = kpvis2w

    desc_endpoints['xy_maps1to2'] = xy_maps1to2
    desc_endpoints['visible_masks1'] = visible_masks1
    desc_endpoints['apn_patches'] = apn_patches
    desc_endpoints['neg_inds'] = neg_inds

    #----------------------------------
    #  Training Loss
    #----------------------------------
    final_det_loss = det_loss + config.weight_det_loss * desc_pair_loss + config.ori_weight * ori_loss + config.scale_weight * scale_loss
    final_desc_loss = desc_loss
    tf.summary.scalar('final_det_loss', final_det_loss)
    tf.summary.scalar('final_desc_loss', final_desc_loss)

    det_endpoints['loss'] = final_det_loss

    #----------------------------------
    #  Evaluation of Descriptor (make sure the following code only works if batch_size=1)
    #----------------------------------

    eval_endpoints = build_matching_estimation(config, desc_feats1, desc_feats2, 
                                                        kpts1, kpts2,
                                                        kpts2w, kpvis2w, dist_thresh=config.match_reproj_thresh)
    sift_endpoints = build_competitor_matching_estimation(config, dist_thresh=config.match_reproj_thresh)

    return final_det_loss, final_desc_loss, det_endpoints, desc_endpoints, eval_endpoints, sift_endpoints
    # return loss, loss_det, det_endpoints, desc_endpoints, eval_endpoints, sift_endpoints
