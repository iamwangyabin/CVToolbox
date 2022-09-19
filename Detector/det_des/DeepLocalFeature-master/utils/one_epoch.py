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

LOCAL_PATH = '../'
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)

from DatasetsBuilder import *

from det_tools import *
# from eval_tools import compute_sift, compute_sift_multi_scale, draw_match, draw_keypoints, draw_match2
from common.tf_layer_utils import *
from common.tf_train_utils import get_optimizer, get_piecewise_lr, get_activation_fn
from common.tfvisualizer import log_images, convert_tile_image

from inference import *
import getConfig

MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)


def epoch():
    config,unparsed=getConfig.get_lfconfig()
    tf.reset_default_graph() # for sure
    # set_summary_visibility(variables=False, gradients=False)

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
    va_iter_list = [va.make_initializable_iterator() for va in va_dataset_list]

    is_training_ph = tf.placeholder(tf.bool, shape=(), name='is_training')

    psf = tf.constant(get_gauss_filter_weight(config.hm_ksize, config.hm_sigma)[:,:,None,None], dtype=tf.float32) 
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step2 = tf.Variable(0, name='global_step2', trainable=False)
