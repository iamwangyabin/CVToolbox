import os
import sys
import numpy as np
from common.argparse_utils import *

def get_lfconfig():
    parser = get_parser()
    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                                help='the number of threads (for dataset)')
    train_arg = add_argument_group('Train', parser)
    train_arg.add_argument('--log_dir', type=str, default='logs/',help='where to save')
    train_arg.add_argument('--pretrain_dir', type=str, default='',
                            help='pretrain model directory')
    train_arg.add_argument('--clear_logs', action='store_const',
                            const=True, default=False,
                            help='clear logs if it exists')
    train_arg.add_argument('--show_histogram', action='store_const',
                            const=True, default=False,
                            help='show variable / gradient histograms on tensorboard (consume a lot of disk space)')
    train_arg.add_argument('--max_itr', type=int, default=50000,
                            help='max epoch')
    train_arg.add_argument('--batch_size', type=int, default=6,
                            help='batch size')
    train_arg.add_argument('--optim_method', type=str, default='Adam',
                            help='adam, momentum, ftrl, rmsprop')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
    train_arg.add_argument('--lr_decay', type=str2bool, default=False,
                            help='apply lr decay')
    det_train_arg = add_argument_group('Detector Train', parser)
    det_train_arg.add_argument('--det_loss', type=str, default='l2loss',
                            help='l2loss|lift')
    det_train_arg.add_argument('--hm_ksize', type=int, default=15,
                            help='gauss kernel size for heatmaps (odd value)')
    det_train_arg.add_argument('--hm_sigma', type=float, default=0.5,
                            help='gauss kernel sigma for heatmaps')
    det_train_arg.add_argument('--nms_thresh', type=float, default=0.0,
                            help='threshold before non max suppression')
    det_train_arg.add_argument('--nms_ksize', type=int, default=5,
                            help='filter size of non max suppression')
    det_train_arg.add_argument('--top_k', type=int, default=512,
                            help='select top k keypoints')
    det_train_arg.add_argument('--weight_det_loss', type=float, default=0.01,
                            help='L_det = L2-score map + lambda * pairwise-loss')
    # supervised orientation training
    det_train_arg.add_argument('--ori_loss', type=str, default='l2loss',
                            help='orientation loss (l2loss|cosine)')
    det_train_arg.add_argument('--ori_weight', type=float, default=0.1,
                            help='orientation weight (L_det = L_score + L_ori + L_scale + L_pair)')
    det_train_arg.add_argument('--scale_weight', type=float, default=0.1,
                            help='scale weight (L_det = L_score + L_ori + L_scale + L_pair)')
    desc_train_arg = add_argument_group('Descriptor Train', parser)
    desc_train_arg.add_argument('--desc_loss', type=str, default='triplet',
                            help='descriptor loss')
    desc_train_arg.add_argument('--desc_margin', type=float, default=1.0,
                            help='triplet margin for descriptor loss')
    desc_train_arg.add_argument('--crop_radius', type=int, default=16, 
                            help='crop radius of region proposal')
    desc_train_arg.add_argument('--patch_size', type=int, default=32, 
                            help='cropped patch size')
    desc_train_arg.add_argument('--mining_type', type=str, default='rand_hard_sch',
                            help='negative mining type (hard|random|hard2geom|rand_hard_sch)')
    desc_train_arg.add_argument('--desc_inputs', type=str, default='photos',
                            help='descriptor inputs type (det_feats|photos|concat)')
    desc_train_arg.add_argument('--desc_train_delay', type=int, default=0,
                            help='starting iteration to train descriptor')
    dataset_arg = add_argument_group('Dataset', parser)
    dataset_arg.add_argument('--dataset', type=str, default='sfm',
                            help='dataset (scenenet|scannet)')
    dataset_arg.add_argument('--sfm_img_dir', type=str, default='./release/outdoor_examples/images',
                            help='sfm image root directory')
    dataset_arg.add_argument('--sfm_dpt_dir', type=str, default='./release/outdoor_examples/depths',
                            help='sfm depth and pose root directory')
    dataset_arg.add_argument('--sfm_seq', type=str, default='sacre_coeur',
                            help='sfm sequence name. concatenate with , if you want to add multiple sequences')
    dataset_arg.add_argument('--rot_aug', type=str2bool, default=True,
                            help='add rotation augmentation')
    dataset_arg.add_argument('--scale_aug', type=str2bool, default=True,
                            help='add rotation augmentation')
    dataset_arg.add_argument('--aug_max_degree', type=int, default=180,
                            help='max degree for rot, min_degree will be decided by -max_degree')
    dataset_arg.add_argument('--aug_max_scale', type=float, default=1.414,
                            help='max scale (in linear space, min_scale and max_scale should be symmetry in log-space)')
    dataset_arg.add_argument('--data_raw_size', type=int, default=362,
                            help='image raw size')
    dataset_arg.add_argument('--data_size', type=int, default=256,
                            help='image size (data_size * sqrt(2) = data_raw_size)')
    dataset_arg.add_argument('--depth_thresh', type=float, default=1.0,
                            help='depth threshold for inverse warping')
    dataset_arg.add_argument('--match_reproj_thresh', type=float, default=5,
                            help='matching reprojection error threshold')
    det_net_arg = add_argument_group('Detector CNN', parser)
    det_net_arg.add_argument('--detector', type=str, default='mso_resnet_detector',
                            help='network model (mso_resnet_detector)')
    det_net_arg.add_argument('--activ_fn', type=str, default='leaky_relu',
                            help='activation function (relu|leaky_relu|tanh)')
    det_net_arg.add_argument('--leaky_alpha', type=float, default=0.2,
                            help='alpha of leaky relu')
    det_net_arg.add_argument('--perform_bn', type=str2bool, default=True,
                            help='use batch normalization')
    det_net_arg.add_argument('--net_channel', type=int, default=16,
                            help='init network channels')
    det_net_arg.add_argument('--net_block', type=int, default=3,
                            help='# residual block (each block has 2 conv)')    
    det_net_arg.add_argument('--conv_ksize', type=int, default=5,
                            help='kernel size of conv layer')
    det_net_arg.add_argument('--ori_ksize', type=int, default=5,
                            help='kernel size of orientation conv layer')
    det_net_arg.add_argument('--sm_ksize', type=int, default=15,
                            help='kernel size of spatial softmax')    
    det_net_arg.add_argument('--com_strength', type=float, default=3.0,
                            help='center of the mass')
    det_net_arg.add_argument('--train_ori', type=str2bool, default=True,
                            help='train ori params')
    det_net_arg.add_argument('--net_min_scale', type=float, default=1.0/np.sqrt(2),
                            help='min scale at pyramid heatmaps')
    det_net_arg.add_argument('--net_max_scale', type=float, default=np.sqrt(2),
                            help='max scale at pyramid heatmaps')
    det_net_arg.add_argument('--net_num_scales', type=int, default=5,
                            help='number of scale maps (e.g. num_scales = (log2(1)-log2(min_scale)) / log2(2**(1/3)) )')
    desc_net_arg = add_argument_group('Descriptor CNN', parser)
    desc_net_arg.add_argument('--descriptor', type=str, default='simple_desc',
                            help='descriptor network model (simple_desc)')
    desc_net_arg.add_argument('--desc_activ_fn', type=str, default='relu',
                            help='activation function (relu|leaky_relu|tanh)')
    desc_net_arg.add_argument('--desc_leaky_alpha', type=float, default=0.2,
                            help='alpha of leaky relu')
    desc_net_arg.add_argument('--desc_perform_bn', type=str2bool, default=True,
                            help='use batch normalization')
    desc_net_arg.add_argument('--desc_net_channel', type=int, default=64,
                            help='init network channels')
    desc_net_arg.add_argument('--desc_net_depth', type=int, default=3,
                            help='# conv layers')    
    desc_net_arg.add_argument('--desc_conv_ksize', type=int, default=3,
                            help='kernel size of conv layer')    
    desc_net_arg.add_argument('--desc_norm', type=str, default='l2norm',
                            help='feature normalization (l2norm|inst|rootsift|non)')    
    desc_net_arg.add_argument('--desc_dim', type=int, default=256,
                            help='descriptor feature dimension')
    misc_arg = add_argument_group('Misc.', parser)
    misc_arg.add_argument('--train_same_time', type=str2bool, default=True,
                            help='train det loss and ori loss at the same time')
    misc_arg.add_argument('--input_inst_norm', type=str2bool, default=True,
                            help='input are normalized with inpstance norm')
    misc_arg.add_argument('--hard_geom_thresh', type=str2bool, default=32,
                            help='x,y coordinate distance threshold')
    misc_arg.add_argument('--init_num_mine', type=int, default=64,
                            help='initial top-k sampling for negative mining')
    desc_train_arg.add_argument('--min_num_pickup', type=int, default=5,
                            help='minimum random pickup')
    desc_train_arg.add_argument('--pickup_delay', type=float, default=0.9,
                            help='decay rate in every 1000 iteration')
    misc_arg.add_argument('--soft_scale', type=str2bool, default=True,
                            help='make scale differentiable')
    misc_arg.add_argument('--soft_kpts', type=str2bool, default=True,
                            help='make delta xy differentiable')
    misc_arg.add_argument('--do_softmax_kp_refine', type=str2bool, default=True,
                            help='do softmax again for kp refinement')
    misc_arg.add_argument('--kp_loc_size', type=int, default=9,
                            help='make scale differentiable')
    misc_arg.add_argument('--score_com_strength', type=float, default=100,
                            help='com strength')
    misc_arg.add_argument('--scale_com_strength', type=float, default=100,
                            help='com strength')
    misc_arg.add_argument('--kp_com_strength', type=float, default=1.0,
                            help='com strength')
    misc_arg.add_argument('--use_nms3d', type=str2bool, default=True,
                            help='use NMS3D to detect keypoints')

    config, unparsed = get_config(parser)

    return config,unparsed