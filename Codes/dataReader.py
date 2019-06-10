import numpy as np
import os
import sys
from easydict import EasyDict as edict
import tensorflow as tf
import scipy.io as io

from tensorflow.python.framework import ops

from data_preprocessing import DataPreprocess
from utils import replace_mask_with_value

def data_mirroring(data):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    data = tf.reverse(data, mirror)
    
    return data

class DataReader(object):
    '''Generic DataReader which reads images and depth from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_home, dataset_name, tag_list, params,coord,
                  random_scale = True, random_mirror = True,random_crop = True, gt_type = 'fused', 
                  shuffle_flag = True, data_type = 1, dataset_catgry = 1, dataoption = 'color_dc_mask',queue_flag = True, num_preprocess_threads = 1):

        self.data_home = data_home
        self.dataset_name = dataset_name
        self.tag_list = tag_list        
        self.channelinfo = edict({})
        self.input_size = params['input_size']        
        self.coord = coord
        self.queue_flag = queue_flag
        self.num_preprocess_threads = num_preprocess_threads
        self.datainput = dataoption
        self.gt_type = gt_type
        self.dataset_catgry = dataset_catgry
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.params = params        
        self.shuffle_flag = shuffle_flag
        self.data_type = data_type
        #Please Check Flag for Interpolation
        images_list, depthgt_list, subsample_list = self.read_labeled_data_list(gt_type = self.gt_type)

        self.imagelist_tf = ops.convert_to_tensor(images_list, dtype=tf.string)
        self.depthgtlist_tf = ops.convert_to_tensor(depthgt_list,dtype = tf.string)
        self.subsamplelist_tf = ops.convert_to_tensor(subsample_list, dtype = tf.string)
        
        #Dataset has several categories.
        # dataset_type = 1; It indicates there are three inputs from hard-drive: color_img,subsampled_depth and gt-depth.
        #      
        
        if self.datainput in ['color_sp_depthgt']:
            #Data-Inputs are Color (RGB) + Sparse Input
            # Labels are Depth-GT 
            self.channelinfo.color_nchannels = 3
            self.channelinfo.depth_nchannels = params['depth_nChannels']  #(only sparse depthchannel)            
            self.channelinfo.label_nchannels = [1]
            self.channelinfo.label_type = ['depth_gt']
            self.channelinfo.data_nchannels = self.channelinfo.color_nchannels + self.channelinfo.depth_nchannels         

        elif self.datainput in ['color_dc_dcclabels']:
            self.channelinfo.color_nchannels = 3
            self.channelinfo.depth_nchannels = params['dce_nChannels'] + 2                      
            self.channelinfo.label_nchannels = [params['dce_nChannels'] + 2, 1]
            self.channelinfo.label_type = ['dcc_labelgt','cont_labelgt']
            self.channelinfo.data_nchannels = self.channelinfo.color_nchannels + self.channelinfo.depth_nchannels                           

        elif self.datainput == 'color_sp_dcclabels':
            self.channelinfo.color_nchannels = 3
            self.channelinfo.depth_nchannels = params['depth_nChannels']              
            self.channelinfo.label_nchannels = [params['dce_nChannels'] + 2, 1]
            self.channelinfo.label_type = ['dcc_labelgt','cont_labelgt']
            self.channelinfo.data_nchannels = self.channelinfo.color_nchannels + self.channelinfo.depth_nchannels
        

        if self.queue_flag:
            self.queue = tf.train.slice_input_producer([self.imagelist_tf, self.depthgtlist_tf, self.subsamplelist_tf],
                                                   shuffle = self.shuffle_flag) # not shuffling if it is val
        else:
            self.queue = tf.stack([self.imagelist_tf, self.depthgtlist_tf, self.subsamplelist_tf])


    def read_labeled_data_list(self,gt_type = 'fused'):

        
        color_imgs = []
        depthgt_labels = []
        subsample_depths = []
        

        for tag in self.tag_list:
            color_img = tag['color_tag']

            
            
            if not tf.gfile.Exists(color_img):
                raise ValueError('Failed to find file: ' + color_img)    
            
            color_imgs.append(color_img)    
            
            subsample_depth = tag['sparsesubsample_tag']           
            
            
            if not tf.gfile.Exists(subsample_depth):
                raise ValueError('Failed to find file: ' + subsample_depth)
            
            subsample_depths.append(subsample_depth)             

            if gt_type == 'fused':
                depthgt_label = tag['fusedgt_tag']
            elif gt_type == 'annotated':    
                depthgt_label = tag['annotatedgt_tag']            
            else:
                ValueError('Unrecognized gt type. Exiting ...\n')
            

            if not tf.gfile.Exists(depthgt_label):
                raise ValueError('Failed to find file: ' + depthgt_label)

            depthgt_labels.append(depthgt_label)            
            

        return color_imgs, depthgt_labels, subsample_depths

    def read_data_from_disk(self, queue): # optional pre-processing arguments
        
        dataproc = DataPreprocess(queue, coords = self.coord, dataset_catgry = self.dataset_catgry, dataset_type = self.data_type)
        
        h,w = self.params['input_size']
        dstep = self.params['dce_dstep']        
        
        if self.data_type == 1:

            color_img = dataproc.load_colorimg(height = h, width = w, channels = 3)            
            depth_gt = dataproc.load_depthgt(height = h, width = w, channels = 1)            
            subsample_depth = dataproc.load_subsampledepth(height = h, width = w, channels = 1)
            
            if self.params['orig_normalizefac']:
                depth_gt = tf.cast(depth_gt,tf.float32)*100/256
                subsample_depth = tf.cast(subsample_depth,tf.float32)*100/256
            else:
                depth_gt = tf.cast(depth_gt,tf.float32)
                subsample_depth = tf.cast(subsample_depth,tf.float32)                
            
            color_img = tf.slice(color_img,[self.params['truncated_height_start'], 0, 0],[self.params['truncated_height_end'], w, 3])
            depth_gt = tf.slice(depth_gt,[self.params['truncated_height_start'], 0,0],[self.params['truncated_height_end'], w, 1])
            subsample_depth = tf.slice(subsample_depth, [self.params['truncated_height_start'], 0, 0], [self.params['truncated_height_end'],w,1])
        

        if self.datainput in ['color_dc_dcclabels']:
            color_img_processed = dataproc.preprocess_color(color_img, self.params['coloraugmentflag'])
            depth_gt = tf.cast(depth_gt,tf.float32)           
            
            if self.params['Gen_uniformsampflag']:
                subsample_depth = dataproc.uniform_sampling(depth_gt, self.params['Uniform_samp'])
            

            
            subsampledepth_dcc = dataproc.depth_2_dcc_channelsgeneralize(subsample_depth, dstep, 
                                    self.params['depth_maxrange'], spatial_dim = (self.params['truncated_height_end'], w))
            depth_gt_dcc = dataproc.depth_2_dcc_channelsgeneralize(depth_gt, self.params['dce_dstep'], self.params['depth_maxrange'], spatial_dim = (self.params['truncated_height_end'], w))
            #depth_gt_dcc = dataproc.depth_2_dcc_channelsgeneralize(depth_gt, dstep, self.params['depth_maxrange'], oorFlag = False, spatial_dim = (h,w))
            depth_gt_dcc = tf.squeeze(depth_gt_dcc)

            data_processed = tf.concat([color_img_processed, subsampledepth_dcc, depth_gt_dcc, depth_gt], axis = 2)                
        
        else:
            ValueError('Data-Input Type is Unrecognized. Exiting ...\n')
        
        if self.random_mirror:
            data_processed = data_mirroring(data_processed)

        if self.random_crop:
            data, labels = self.random_crop_and_pad_data_and_labels(data_processed, self.params['crop_size'][0], self.params['crop_size'][1])
        else:
            data, labels = self.crop_pad_data_labels(data_processed, self.params['crop_size'][0], self.params['crop_size'][1])

        return data, labels    


    def random_crop_and_pad_data_and_labels(self, combined, crop_h, crop_w, ignore_label = 255):
    
        data_shape = tf.shape(combined)
        combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, data_shape[0]), tf.maximum(crop_w, data_shape[1]))

        data_nchannels = self.channelinfo.data_nchannels    
        
        label_cumchannels = np.zeros((len(self.channelinfo.label_nchannels) + 1),dtype = np.int16)

        for indx in range(0,len(self.channelinfo.label_nchannels)):
            label_cumchannels[indx + 1] = np.sum(self.channelinfo.label_nchannels[:indx + 1])
        
        combined_crop = tf.random_crop(combined_pad,[crop_h,crop_w,data_nchannels + np.sum(self.channelinfo.label_nchannels)])       
            

        data_crop = combined_crop[:, :, :data_nchannels]
        labels_crop = [combined_crop[:, :, data_nchannels + label_cumchannels[indx] : data_nchannels + label_cumchannels[indx] + self.channelinfo.label_nchannels[indx]] 
                    for indx in range(len(self.channelinfo.label_nchannels))]
        
        labels_crop = tf.concat(labels_crop, axis = 2)
        
        

        return data_crop, labels_crop

    def crop_pad_data_labels(self, combined, crop_h, crop_w ):

        data_shape = tf.shape(combined)
        combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, data_shape[0]), tf.maximum(crop_w, data_shape[1]))

        data_nchannels = self.channelinfo.data_nchannels    
        
        label_cumchannels = np.zeros((len(self.channelinfo.label_nchannels) + 1),dtype = np.int16)

        for indx in range(0,len(self.channelinfo.label_nchannels)):
            label_cumchannels[indx + 1] = np.sum(self.channelinfo.label_nchannels[:indx + 1])

        act_width = combined_pad.shape.as_list()[1]

        ntiles = np.uint8(np.ceil(act_width/crop_w))
        #mu = np.round(ntiles/2)
        #sigma = 1
        #nval = np.random.normal(0,1)

        #sample = nval*sigma + mu
        #sample = np.min([sample, ntiles - 1])
        #sample = np.max([sample, 0])

        #offset_width = int(np.random.randint(ntiles))*crop_w    
        data_crop = []
        labels_crop = []
        for indx in range(ntiles):
            offset_width = indx*crop_w
            if (offset_width + crop_w - 1) > act_width:
                offset_width = act_width - crop_w
        
            combined_crop = tf.image.crop_to_bounding_box(combined, 0, offset_width, crop_h, crop_w)
            #combined_crop = tf.image.crop_to_bounding_box(combined,0, 962, 280, 280)

            data_crop.append(combined_crop[..., :data_nchannels])
            labels_crop_temp = [combined_crop[..., data_nchannels + label_cumchannels[indx] : data_nchannels + label_cumchannels[indx] + self.channelinfo.label_nchannels[indx]] 
                    for indx in range(len(self.channelinfo.label_nchannels))]
        
            labels_crop.append(tf.concat(labels_crop_temp, axis = 2))

        return data_crop, labels_crop

        

    def dequeue(self, num_elements):
        
        data, labels = self.read_data_from_disk(self.queue)        
        
        data_batch, label_batch = tf.train.batch([data, labels], num_elements,num_threads = self.num_preprocess_threads)
        
        
        return data_batch, label_batch
