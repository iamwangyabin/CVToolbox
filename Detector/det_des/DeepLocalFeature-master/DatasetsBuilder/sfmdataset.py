import numpy as np
import os
import random
import sys
import tensorflow as tf

from .dataset_tools import *

# def get_delta_pose(C1TW,C2TW):
#     C1TW_R = C1TW[:3,:3]
#     C2TW_R = C2TW[:3,:3]
#     C1TW_t = np.expand_dims(C1TW[:3,3], axis=1)
#     C2TW_t = np.expand_dims(C2TW[:3,3], axis=1)
#     ones=np.array([0.,0.,0.,1.])
#     R=(np.linalg.inv(C2TW_R)).dot(C1TW_R)
#     t=(np.linalg.inv(C2TW_R)).dot(C1TW_t-C2TW_t)
#     T=np.hstack((R,t))
#     T=np.vstack((T,ones))
#     T=np.linalg.inv(T)
#     T_=np.linalg.inv(T)
#     return T,T_
    
def get_delta_pose(c1Tw, c2Tw):
    # cTw = world to camera pose [4x4 matrix]
    # return = c2Tc1, which means c1 to c2 pose
    c1Rw = tf.slice(c1Tw, [0,0], [3,3])
    c2Rw = tf.slice(c2Tw, [0,0], [3,3])
    c1Pw = tf.slice(c1Tw, [0,3], [3,1])
    c2Pw = tf.slice(c2Tw, [0,3], [3,1])
    wPc1 = -tf.matmul(c1Rw, c1Pw, transpose_a=True) # wPc = -wRc cPw
    wPc2 = -tf.matmul(c2Rw, c2Pw, transpose_a=True) # wPc = -wRc cPw
    c2Rc1 = tf.matmul(c2Rw, c1Rw, transpose_b=True) # c2Rc1 = c2Rw wRc1
    c2Pc1 = tf.matmul(c2Rw, wPc1-wPc2) # c2Pc1 = c2Rw (wPc1-wPc2)
    # c2Tc1 (4x4) = 
    # | c2Rc1 c2Pc1 |
    # |   0     1   |
    c2Tc1 = tf.concat([c2Rc1, c2Pc1], axis=1)
    c2Tc1 = tf.concat([c2Tc1, tf.constant([[0,0,0,1]], dtype=tf.float32)], axis=0)
    c1Tc2 = tf.matrix_inverse(c2Tc1)
    return c2Tc1, c1Tc2


# rootdir='../dataset/'
# render_path=['scan']


class SfMDataset(object):
    def __init__(self, out_size=(320, 320), warp_aug_mode='none', flip_pair=False, max_degree=180, max_scale=np.sqrt(2), min_scale=None, compress=False, num_threads=8):
        self.num_threads = num_threads
        self.out_size = out_size # [height, width]
        self.warp_aug_mode = warp_aug_mode
        # self.warp_aug_mode = 'none'
        # print('Disable warp_aug_mode @ SfMDataset')
        self.compression_type = 'GZIP' if compress else None

        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.depth_thresh = 10.0
        self.flip_pair = flip_pair
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_degree = max_degree

    def get_dataset(self, root_dir, imroot_dir, render_paths, phase, batch_size=32, shuffle=True, num_epoch=None, seed=None, max_examples=-1):
        # 不知道这
        # table_dir = os.path.join(root_dir, '../../../scannet/params/')
        self.random_transformer = RandomTransformer('table_dir', self.warp_aug_mode, max_scale=self.max_scale, min_scale=self.min_scale, max_degree=self.max_degree)

        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        
        if not root_dir.endswith('/'):
            root_dir += '/'
        self.root_dir = tf.convert_to_tensor(root_dir)

        # if not imroot_dir.endswith('/'):
        #     imroot_dir += '/'
        # self.imroot_dir = tf.convert_to_tensor(imroot_dir)
        # root_dir, render, pose_fname
        # print()

        # pass
        pose_tfrecords = []
        total_num_photos = 0
        for render in render_paths:
            if phase == 'train':
                # pose_fname = 'train_{}.tfrecord'.format(max_examples)
                # size_fname = None
                # print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                pose_fname = 'train.tfrecord'
                size_fname = None
            elif phase == 'valid':
                pose_fname = 'valid.tfrecord'
                size_fname = 'valid_size.txt'
            else:
                print("Give a phase")
                pass
            pose_tfrecords.append(os.path.join(root_dir, render, pose_fname))
            if size_fname is None:
                size = 1674
            else:
                with open(os.path.join(root_dir, render, size_fname)) as f:
                    size = int(f.readline())
                print('{} has {} examples'.format(render, size))
                if max_examples > 0:
                    size = min(size, max_examples)
                    print('---> actual size={}'.format(size))
            total_num_photos += size
            print(pose_tfrecords)
        self.total_num_photos = total_num_photos
        self.num_photos_per_seq_data = np.array([total_num_photos], dtype=np.int32)

        dataset = tf.data.TFRecordDataset(pose_tfrecords, compression_type=self.compression_type)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.total_num_photos, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(batch_size)
        return dataset

    def parser(self, serialized):
        with tf.name_scope('parse_example'):
            example = tf.parse_single_example(serialized, features={
                'rgb1_filename': tf.FixedLenFeature([], tf.string),
                'rgb2_filename': tf.FixedLenFeature([], tf.string),
                'depth1_filename': tf.FixedLenFeature([], tf.string),
                'depth2_filename': tf.FixedLenFeature([], tf.string),
                'shape1': tf.FixedLenFeature([2], tf.int64),
                'shape2': tf.FixedLenFeature([2], tf.int64),
                # 'bbox1': tf.FixedLenFeature([4], tf.int64), # x1,x2,y1,y2
                # 'bbox2': tf.FixedLenFeature([4], tf.int64),
                'c1Tw': tf.FixedLenFeature([16], tf.float32),
                'c2Tw': tf.FixedLenFeature([16], tf.float32),
                'K1': tf.FixedLenFeature([9], tf.float32),
                'K2': tf.FixedLenFeature([9], tf.float32),
            })

        # Flip images
        if self.flip_pair:
            # pair is always idx1 < idx2 so that it will be effective to switch pairs randomly
            flip_example = {
                'rgb1_filename': example['rgb2_filename'],
                'rgb2_filename': example['rgb1_filename'],
                'depth1_filename': example['depth2_filename'],
                'depth2_filename': example['depth1_filename'],
                'shape1': example['shape2'],
                'shape2': example['shape1'],
                # 'bbox1': example['bbox2'], 
                # 'bbox2': example['bbox1'],
                'c1Tw': example['c2Tw'],
                'c2Tw': example['c1Tw'],
                'K1': example['K2'],
                'K2': example['K1'],
            }
            is_flip = tf.less_equal(tf.random_uniform([]), 0.5)
            example = tf.cond(is_flip, lambda: flip_example, lambda: example)            

        shape1 = example['shape1']
        shape2 = example['shape2']
        c1Tw = tf.reshape(example['c1Tw'], [4,4])
        c2Tw = tf.reshape(example['c2Tw'], [4,4])
        K1 = tf.reshape(example['K1'], [3,3])
        K2 = tf.reshape(example['K2'], [3,3])
        # bb1 = example['bbox1']
        # bb2 = example['bbox2']
        rgb1_filename = example['rgb1_filename']
        rgb2_filename = example['rgb2_filename']
        depth1_filename = example['depth1_filename']
        depth2_filename = example['depth2_filename']

        rgb1 = self._decode_rgb(rgb1_filename, shape1)
        rgb2 = self._decode_rgb(rgb2_filename, shape2)
        depth1, valid_mask1 = self._decode_depth(depth1_filename, shape1)
        depth2, valid_mask2 = self._decode_depth(depth2_filename, shape2)

        dv1 = tf.concat([depth1, valid_mask1], axis=-1)
        dv2 = tf.concat([depth2, valid_mask2], axis=-1)
        
        if self.out_size is not None:
            sy1 = float(self.out_size[0]) / tf.to_float(tf.shape(rgb1)[0])
            sx1 = float(self.out_size[1]) / tf.to_float(tf.shape(rgb1)[1])
            sy2 = float(self.out_size[0]) / tf.to_float(tf.shape(rgb2)[0])
            sx2 = float(self.out_size[1]) / tf.to_float(tf.shape(rgb2)[1])
            S1 = make_scale_theta(sx1, sy1)
            S2 = make_scale_theta(sx2, sy2)
            K1 = tf.matmul(S1, K1)
            K2 = tf.matmul(S2, K2)
            
            # do not use linear interpolation on depth and valid_masks
            rgb1 = tf.image.resize_images(rgb1, (self.out_size[0],self.out_size[1]))
            dv1 = tf.image.resize_images(dv1, (self.out_size[0],self.out_size[1]),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            rgb2 = tf.image.resize_images(rgb2, (self.out_size[0],self.out_size[1]))
            dv2 = tf.image.resize_images(dv2, (self.out_size[0],self.out_size[1]),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        depth1 = tf.slice(dv1, [0,0,0], [-1,-1,1])        
        valid_mask1 = tf.slice(dv1, [0,0,1], [-1,-1,1])        
        depth2 = tf.slice(dv2, [0,0,0], [-1,-1,1])        
        valid_mask2 = tf.slice(dv2, [0,0,1], [-1,-1,1])        
        # return rgb1_filename, rgb2_filename, c1Tw, c2Tw
        # Pose
        c2Tc1, c1Tc2 = get_delta_pose(c1Tw, c2Tw)
        # return rgb1_filename, rgb2_filename, c2Tc1, c1Tc2, c1Tw, c2Tw

        # get random thetas (doesnot support table-random)
        theta_params, use_augs = self.random_transformer.get_theta_params(None)
        # with tf.Session() as sess: 
        #     rgb1_filename2= sess.run(rgb1_filename)
        # print(rgb1_filename2)
        # add in-plane rotation
        intheta_c2Rc1 = tf.py_func(get_inplane_rotation, [c2Tc1[:3,:3]], [tf.float32])
        intheta_c1Rc2 = tf.py_func(get_inplane_rotation, [c1Tc2[:3,:3]], [tf.float32])
        theta_params = tf.concat([theta_params, intheta_c2Rc1, intheta_c1Rc2], axis=0)

        return rgb1, rgb2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_augs
        
    def _decode_rgb(self, filename, shape):
        rgb = tf.read_file(filename)
        rgb = tf.image.decode_jpeg(rgb, 1)
        rgb = tf.cast(rgb, tf.float32) / 255.0
        return rgb
    
    def _decode_depth(self, filename, shape):
        depth = tf.read_file(filename)
        depth = tf.image.decode_png(depth, 1, dtype=tf.uint16) # force to load as grayscale
        depth = tf.scalar_mul(self.depth_factor, tf.cast(depth, tf.float32))
        is_zero = tf.equal(depth, tf.constant(0, dtype=tf.float32))
        valid_mask = tf.cast(tf.logical_not(is_zero), tf.float32)
        far_depth = tf.scalar_mul(self.far_depth_val, tf.ones_like(depth, dtype=tf.float32))
        depth = tf.where(is_zero, far_depth, depth)
        return depth, valid_mask
