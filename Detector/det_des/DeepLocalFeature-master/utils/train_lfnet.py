#!/usr/bin/env python

# Self Supervised Joint Training for Keypoints Detector & Descriptor

from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import time
import cv2
from tqdm import tqdm
import pickle

# LOCAL_PATH = '../'
# if LOCAL_PATH not in sys.path:
#     sys.path.append(LOCAL_PATH)
sys.path.append("../")
# from DatasetsBuilder import *
import DatasetsBuilder
# from datasets.scenenet import SceneNetPairwiseDataset
# from datasets.se3dataset import SE3PairwiseDataset

from det_tools import *
from eval_tools import compute_sift, compute_sift_multi_scale, draw_match, draw_keypoints, draw_match2
from common.tf_layer_utils import *
from common.tf_train_utils import get_optimizer, get_piecewise_lr, get_activation_fn
from common.tfvisualizer import log_images, convert_tile_image

from inference import *
from buildTrainNetwork import build_training_network

MODEL_PATH = '../models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

g_sift_metrics = {
    # 'default': [None] * 100,
    # 'i_hpatches': [None] * 100,
    # 'v_hpatches': [None] * 100,
}

SAVE_MODEL = True

def main(config):
    tf.reset_default_graph() # for sure
    set_summary_visibility(variables=False, gradients=False)

    log_dir = config.log_dir
    batch_size = config.batch_size
    optim_method = config.optim_method
    learning_rate = config.lr
    va_batch_size = 1

    tr_loader = DatasetsBuilder.SfMDataset(out_size=(config.data_raw_size, config.data_raw_size), 
                    warp_aug_mode='random', flip_pair=True, max_degree=config.aug_max_degree, max_scale=config.aug_max_scale,
                    num_threads=config.num_threads)
    tr_dataset = tr_loader.get_dataset('../dataset', 'config.sfm_img_dir', 
                    'scan', phase='train',
                    batch_size=batch_size, shuffle=True)

    config.depth_thresh = tr_loader.depth_thresh
    print('Reset depth_thresh: {}, it may be better to use placeholder'.format(config.depth_thresh))

    # use feedable iterator to switch training / validation dataset without unnecessary initialization
    handle = tf.placeholder(tf.string, shape=[])
    dataset_iter = tf.data.Iterator.from_string_handle(handle, tr_dataset.output_types, tr_dataset.output_shapes) # create mock of iterator
    next_batch = list(dataset_iter.get_next()) #tuple --> list to make it possible to modify each elements

    tr_iter = tr_dataset.make_one_shot_iterator() # infinite loop

    is_training_ph = tf.placeholder(tf.bool, shape=(), name='is_training')

    psf = tf.constant(get_gauss_filter_weight(config.hm_ksize, config.hm_sigma)[:,:,None,None], dtype=tf.float32) 
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step2 = tf.Variable(0, name='global_step2', trainable=False)

    # Euclidean transformation data augmentation
    next_batch = DatasetsBuilder.euclidean_augmentation(next_batch, (config.data_size, config.data_size), config.rot_aug, config.scale_aug)

    det_loss, desc_loss, det_endpoints, desc_endpoints, eval_endpoints, sift_endpoints = \
                        build_training_network(config, next_batch, is_training_ph, psf, global_step)
    # var_list = det_endpoints['var_list'] + desc_endpoints['var_list']
    det_var_list = det_endpoints['var_list'] + det_endpoints['mso_var_list']
    desc_var_list = desc_endpoints['var_list']

    if config.lr_decay:
        boundaries = [5000, 15000, 30000, 50000]
        lr_levels = [0.1**i for i in range(len(boundaries))]
        lr_values = [learning_rate * decay for decay in lr_levels]
        learning_rate = get_piecewise_lr(global_step, boundaries, lr_values, show_summary=True)
        print('Enable adaptive learning. LR will decrease {} when #iter={}'.format(lr_values, boundaries))        

    # We should rename desc_minimize_op as desc_minimizer and so on.
    # descriptor minimizer
    desc_minimize_op = get_optimizer(optim_method, global_step, learning_rate, desc_loss, desc_var_list, show_var_and_grad=config.show_histogram)
    # detector minimizer
    det_minimize_op = get_optimizer(optim_method, global_step, learning_rate, det_loss, det_var_list, show_var_and_grad=config.show_histogram)

    print('Done.')

    # Create a session
    print('Create & Initialize session...')

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True # almost the same as tf.InteractiveSession
    sess = tf.Session(config=tfconfig)

    summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    tr_handle = sess.run(tr_iter.string_handle())
    # va_handle_list = sess.run([va.string_handle() for va in va_iter_list])

    if config.clear_logs and tf.gfile.Exists(log_dir):
        print('Clear all files in {}'.format(log_dir))
        try:
            tf.gfile.DeleteRecursively(log_dir) 
        except:
            printglobal_step2('Fail to delete {}. You probably have to kill tensorboard process.'.format(log_dir))

    # load pretrained model
    if len(config.pretrain_dir) > 0:
        if os.path.isdir(config.pretrain_dir):
            checkpoint = tf.train.latest_checkpoint(config.pretrain_dir)
        else:
            checkpoint = None
        if checkpoint is not None:
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            pretrained_vars = []
            for var in global_vars:
                if 'global_step' in var.name:
                    pass
                else:
                    pretrained_vars.append(var)
            print('Resume pretrained detector...')
            for i, var in enumerate(pretrained_vars):
                print('#{} {} [{}]'.format(i, var.name, var.shape))
            saver = tf.train.Saver(pretrained_vars)
            saver.restore(sess, checkpoint)
            saver = None
            print('Load pretrained model from {}'.format(checkpoint))
        else:
            raise ValueError('Cannot open checkpoint: {}'.format(checkpoint))

    best_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    # latest_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    latest_saver = tf.train.Saver(max_to_keep=100, save_relative_paths=True) # save everything
 
    latest_checkpoint = tf.train.latest_checkpoint(log_dir)
    best_score_filename = os.path.join(log_dir, 'valid', 'best_score.txt')
    best_score = 0 # larger is better
    curr_epoch = 0
    if latest_checkpoint is not None:
        from parse import parse
        print('Resume the previous model...')
        latest_saver.restore(sess, latest_checkpoint)
        curr_step = sess.run(global_step)
        curr_epoch = curr_step // (tr_loader.total_num_photos // batch_size)
        print('Current step={}, epoch={}'.format(curr_step, curr_epoch))
        if os.path.exists(best_score_filename):
            with open(best_score_filename, 'r') as f:
                dump_res = f.read()
            dump_res = parse('{step:d} {best_score:g}\n', dump_res)
            best_score = dump_res['best_score']
            print('Previous best score = {}'.format(best_score))

    train_writer = tf.summary.FileWriter(
        os.path.join(log_dir, 'train'), graph=sess.graph
    )
    valid_writer = tf.summary.FileWriter(
        os.path.join(log_dir, 'valid'), graph=sess.graph
    )    

    if SAVE_MODEL:
        latest_saver.export_meta_graph(os.path.join(log_dir, "models.meta"))
    # Save config
    with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)    

    ops = {
        'is_training': is_training_ph,
        'handle': handle,
        'photos1': next_batch[0],
        'photos2': next_batch[1],
        'depths1': next_batch[2],
        'depths2': next_batch[3],
        'valid_masks1': next_batch[4],
        'valid_masks2': next_batch[5],
        'c2Tc1s': next_batch[6],
        'c1Tc2s': next_batch[7],
        'c1Tws': next_batch[8],
        'c2Tws': next_batch[9],
        'Ks1': next_batch[10],
        'Ks2': next_batch[11],
        'loss': desc_loss,
        'loss_det': det_loss,
        'step': global_step,
        'desc_minimize_op': desc_minimize_op,
        'det_minimize_op': det_minimize_op,
        'global_step': global_step,
        'summary': summary,
    }
    for k, v in det_endpoints.items():
        ops['det_'+k] = v
    for k, v in desc_endpoints.items():
        ops['desc_'+k] = v
    for k, v in eval_endpoints.items():
        ops['eval_'+k] = v
    for k, v in sift_endpoints.items():
        ops['sift_'+k] = v

    #----------------------
    # Start Training
    #----------------------

    num_itr_in_epoch = tr_loader.total_num_photos // batch_size
    save_summary_interval = 200
    save_model_interval = 2000
    valid_interval = 1000

    va_params = {
        'batch_size': va_batch_size,
        'log_dir': log_dir,
        'summary_writer': valid_writer,
        'num_kp': config.top_k,
        'best_score': best_score,
        'best_score_filename': best_score_filename,
        'num_photos_per_seq': None,
        'dataset_size': None,
        'handle': None,
        'ev_init_op': None,
        'best_saver': None,
    }

    # # init g_sift_metrics
    # global g_sift_metrics
    # for attr in va_attributes:
    #     g_sift_metrics[attr['name']] = [None] * 100

    print('Start training.... (1epoch={}itr #size={})'.format(num_itr_in_epoch, tr_loader.total_num_photos))

    def check_counter(counter, interval):
        return (interval > 0 and counter % interval == 0)

    start_itr = sess.run(ops['global_step'])

    for _ in range(start_itr, config.max_itr):
        try:
            
            feed_dict = {
                ops['is_training']: True,
                ops['handle']: tr_handle,
            }


            if config.train_same_time:
                step, _, _ = sess.run([ops['step'], ops['desc_minimize_op'], ops['det_minimize_op']], feed_dict=feed_dict)
            else:
                step, _,  = sess.run([ops['step'], ops['desc_minimize_op']], feed_dict=feed_dict)
                _ = sess.run(ops['det_minimize_op'], feed_dict=feed_dict)
            print('fasdas译者测桑松动的乘啦扫反')
            if check_counter(step, save_summary_interval):
                feed_dict = {
                    ops['is_training']: False,
                    ops['handle']: tr_handle,
                }
                fetch_dict = {
                    'loss': ops['loss'],
                    'loss_det': ops['loss_det'],
                    'det_loss': ops['det_loss'],
                    'desc_loss': ops['desc_loss'],
                    'summary': ops['summary'],
                    'scale_maps': ops['det_scale_maps'],
                }
                start_time = time.time()
                outputs = sess.run(fetch_dict, feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                train_writer.add_summary(outputs['summary'], step) # save summary
                # scale_hist = np.histogram(outputs['scale_maps'], bins=config.net_num_scales, range=[config.net_min_scale, config.net_max_scale])
                # print(scale_hist)
                # print(outputs['scale_maps'].min(), outputs['scale_maps'].max())

                summaries = [tf.Summary.Value(tag='sec/step', simple_value=elapsed_time)]
                train_writer.add_summary(tf.Summary(value=summaries), global_step=step)
                train_writer.flush()

                print('[Train] {}step Loss(desc|det): {:g}|{:g} ({:.3f}|{:.3f}) ({:.1f}sec)'.format(
                            step,
                            outputs['loss'], outputs['loss_det'],
                            outputs['det_loss'], outputs['desc_loss'],
                            elapsed_time))
            if check_counter(step, save_model_interval):
                if SAVE_MODEL and latest_saver is not None:
                    print('#{}step Save latest model'.format(step))
                    latest_saver.save(sess, os.path.join(log_dir, 'models-latest'), global_step=step, write_meta_graph=False)
        except:
            print("一个错误")
        # if check_counter(step, valid_interval):
        #     va_mean_match_score = 0
        #     num_valid_set = 0
        #     for i, va_dataset in enumerate(va_dataset_list):
        #         va_params['num_photos_per_seq'] = va_attributes[i]['num_photos_per_seq']
        #         va_params['dataset_size'] = va_attributes[i]['total_num_photos']
        #         va_params['handle'] = va_handle_list[i]
        #         va_params['ev_init_op'] = va_iter_list[i].initializer

        #         name = va_attributes[i]['name']
        #         if i == 0:
        #             va_params['best_saver'] = best_saver
        #         else:
        #             va_params['best_saver'] = None
        #         print('Eval {} (#samples={})'.format(name, va_params['dataset_size']))
        #         match_score =  eval_one_epoch(sess, ops, va_params, name=name)
        #         if name.startswith('va'):
        #             va_mean_match_score += match_score
        #             num_valid_set += 1
        #     if num_valid_set > 0:
        #         va_mean_match_score /= num_valid_set
        #     if SAVE_MODEL and va_mean_match_score > best_score and best_saver is not None:
        #         best_score = va_mean_match_score
        #         print("Saving best model with valid-score = {}".format(best_score))
        #         best_saver.save(sess, os.path.join(log_dir, 'models-best'), write_meta_graph=False)
        #         with open(best_score_filename, 'w') as f:
        #             f.write('{} {:g}\n'.format(step, best_score))

def overwrite_config(config):
    if config.pretrain_dir == None or len(config.pretrain_dir) == 0:
        print('Skip overwrite config')
        return config# do nothing to overwrite
    pt_config_path = os.path.join(config.pretrain_dir, 'config.pkl')
    if not os.path.exists(pt_config_path):
        print('[WARNING] Not found pretrained config, {}'.format(pt_config_path))
        return config

    with open(pt_config_path, 'rb') as f:
        pt_config = pickle.load(f)

    overwrite_attrs = [
        # Descriptor Train
        'desc_inputs',
        # Detector CNN
        'detector',
        'activ_fn',
        'leaky_alpha',
        'perform_bn',
        'net_channel',
        'net_block',
        'conv_ksize',
        'sm_ksize',
        'com_strength',
        'train_ori',
        'net_min_scale',
        'net_num_scales',
        # Descriptor CNN
        'descriptor',
        'desc_activ_fn',
        'desc_leaky_alpha',
        'desc_perform_bn',
        'desc_net_channel',
        'desc_net_depth',
        'desc_conv_ksize',
        'desc_norm',
        'desc_dim',
    ]
    check_attrs = [
        'hm_ksize',
        'hm_sigma',
        'nms_thresh',
        'nms_ksize',
        'top_k',
        'crop_radius',
        'patch_size',
    ]

    for attr in overwrite_attrs:
        src_val = getattr(config, attr)
        dst_val = getattr(pt_config, attr)
        if src_val != dst_val:
            print('Overwrite {} : {} --> {}'.format(attr, src_val, dst_val))
            setattr(config, attr, dst_val)
    for attr in check_attrs:
        src_val = getattr(config, attr)
        dst_val = getattr(pt_config, attr)
        if src_val != dst_val:
            print('[WARNING] {} has set different values, pretrain={}, now={}'.format(attr, dst_val, src_val))

    # for key, val in sorted(vars(config).items()):
    #     print(f'{key} : {val}')
    print('Finish overwritting config.')
    return config

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=2,
                            help='the number of threads (for dataset)')

    train_arg = add_argument_group('Train', parser)
    train_arg.add_argument('--log_dir', type=str, default='logs/',
                            help='where to save')
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
    train_arg.add_argument('--batch_size', type=int, default=2,
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

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    config = overwrite_config(config)

    if config.aug_max_degree == 0:
        config.rot_aug = False
        config.aug_max_degree = 45
        print('Kill rot_aug because aug_max_degree=0')

    main(config)