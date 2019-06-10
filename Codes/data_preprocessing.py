import numpy as np
import sys
import tensorflow as tf
import scipy.io as io

from utils import prepare_discrete_depthlabel, replace_mask_with_value, sub2ind, ind2sub
#from tensorflow.contrib.layers import dense_to_sparse


class DataPreprocess(object):
    """
    DataProcessor class that handled preprocessing and normalization of the color-channels,
    depth, seglabels etc
    """

    def __init__(self, input_queue = None,coords = None, dataset_type = 1, dataset_catgry = 1, version = 1):

        self.dataset_catgry = dataset_catgry
        self.dataset_type = dataset_type

        self.colorimg_filename = []
        self.depthgt_filename = []
        self.subsampledepth_filename = []        
        self.version = version

        self.coords = coords
        if dataset_type == 1:
            
            self.colorimg_filename = input_queue[0]
            self.depthgt_filename = input_queue[1]
            self.subsampledepth_filename = input_queue[2]           
            

    def load_colorimg(self, height = 113, width = 500,channels = 3):

        if self.colorimg_filename == None:
            RuntimeError('No Image-Name supplied. Exiting ... ')
            sys.exit(0)

        img_contents = tf.read_file(self.colorimg_filename)
        color_img = tf.image.decode_png(img_contents, channels = 3)
        color_img.set_shape((height,width,channels))
        #img_shape = tf.cast(tf.shape(color_img),dtype = tf.float32)
        #img_finshape = tf.cast(img_shape[1:3]*scale,dtype = tf.int32)
        #color_img = tf.image.resize_images(color_img,size = img_finshape,method = tf.image.ResizeMethod.BILINEAR,
        #                              align_corners = True)
        return color_img    
    
    def load_depthgt(self,height = 113, width = 500,channels = 1):

        if self.depthgt_filename == None:
            RuntimeError('No Image-Name supplied. Exiting ...')
            sys.exit(0)

        depth_contents = tf.read_file(self.depthgt_filename)

        if self.dataset_catgry == 1:
            depth = tf.image.decode_png(depth_contents,channels = 3,dtype = tf.uint16)
            depth,__,__ = tf.split(axis = 2, num_or_size_splits=3, value = depth)
        elif self.dataset_catgry == 2 or 3:
             depth = tf.image.decode_png(depth_contents,channels = 1,dtype = tf.uint16)

        depth.set_shape((height, width, channels))                                    
        
        return depth


    def load_subsampledepth(self,height = 113, width = 500, channels = 1):

        if self.subsampledepth_filename == None:
            RuntimeError('No Image-Name supplied. Exiting ...')
            sys.exit(0)

        depth_contents = tf.read_file(self.subsampledepth_filename)
        depth = tf.image.decode_png(depth_contents,channels = 1,dtype = tf.uint16)
        depth.set_shape((height,width, channels))

        return depth   
    
    
    def uniform_sampling(self, depthmap, nsamples):

        mask_keep = tf.greater(depthmap,0)
        n_keep = tf.reduce_sum(tf.cast(mask_keep,tf.float32))            
        prob = nsamples/n_keep

        mask_find = tf.less(tf.random_uniform(tf.shape(depthmap)), prob)

        mask_selected = tf.equal(mask_keep, mask_find)
        indices = tf.where(mask_selected)
        pick_samp = tf.gather_nd(depthmap, indices)

        sampled_depth = tf.scatter_nd(indices, pick_samp, depthmap.shape.as_list())

        return sampled_depth
    
    def preprocess_color(self,color_img,augmentflag = False):

        img_r, img_g, img_b = tf.split(axis = 2, num_or_size_splits = 3, value = color_img)
        color_img = tf.cast(tf.concat(axis = 2, values = [img_b, img_g, img_r]), dtype=tf.float32)
        #Scale the color-img to range value from 0-1
        color_imgprocessed = color_img/255.0

        if augmentflag:

            color_imgprocessed = tf.image.random_brightness(color_imgprocessed, max_delta = 32. / 255.)
            color_imgprocessed = tf.image.random_saturation(color_imgprocessed, lower=0.5, upper=1.5)
            #color_imgprocessed = tf.image.random_hue(color_imgprocessed, max_delta=0.2)
            color_imgprocessed = tf.image.random_contrast(color_imgprocessed, lower=0.5, upper=1.5)
            
            # Subtract off the mean and divide by the variance of the pixels.
            # The random_* ops do not necessarily clamp.
            color_imgprocessed = tf.clip_by_value(color_imgprocessed, 0.0, 1.0)                              
        
        return color_imgprocessed
    
    
    def depth_2_dcc_channelsgeneralize(self, depth, dstep, depth_maxrange,  oormask = None, spatial_dim = (113,500)):
        
        depth = tf.cast(tf.squeeze(depth), tf.float32)
        nChannels = np.int64(depth_maxrange/dstep)

        if oormask is None:
            oormask = tf.zeros(tf.shape(depth)[1:3],tf.bool)

        oormask = tf.squeeze(oormask)
        oor_thresholded = tf.squeeze(depth) > depth_maxrange
        
        oor_completemask = tf.logical_or(tf.cast(oormask,dtype = tf.bool), oor_thresholded)
        oor_completemask = tf.cast(oor_completemask,tf.float32)        
        
        depth_masked = self.replace_mask_with_value(depth, oor_completemask, val_2_replace = depth_maxrange)
        
        oor_completemask = tf.reshape(oor_completemask,[spatial_dim[0], spatial_dim[1], 1])
        
        depth_unfold = tf.reshape(depth_masked,[-1])
        depth_unfold_orig = tf.cast(depth_unfold, tf.float32)/dstep
        depth_unfold_round = tf.round(depth_unfold_orig)
        delta = depth_unfold_orig - depth_unfold_round
        depth_unfold_round = tf.cast(depth_unfold_round,tf.int64)

        good = tf.squeeze(tf.where(tf.logical_and(depth_unfold_round >= 1,depth_unfold_round <= nChannels)))

        depth_unfold_round_filtered = tf.gather(depth_unfold_round,good)

        pix_indices = tf.concat([good,good,good],0)
        depth_bins = tf.concat([depth_unfold_round_filtered - 1, depth_unfold_round_filtered, depth_unfold_round_filtered + 1],0)
        delta_filtered = tf.gather(delta,good)
        
        vals = tf.concat([(0.5 - delta_filtered)/2, 0.5*tf.ones(tf.shape(depth_unfold_round_filtered)), (0.5 + delta_filtered)/2], 0)
        pix_2d = tf.transpose(tf.stack([pix_indices,depth_bins]))
        dcc = tf.sparse_to_dense(pix_2d,[spatial_dim[0]*spatial_dim[1],nChannels + 2], vals,validate_indices = False)
        dcc = tf.reshape(dcc, shape = [spatial_dim[0],spatial_dim[1],nChannels + 2])

        return dcc        

    def depth_2_dcc(self, depth, dstep, nChannels, oormask = None, spatial_dim = (113,500)):

        
        depth = tf.cast(tf.squeeze(depth),tf.float32)

        if oormask is None:
            oormask = tf.zeros(tf.shape(depth)[1:3],tf.bool)

        oormask = tf.squeeze(oormask)
        oor_thresholded = tf.squeeze(depth)/dstep > nChannels
        
        oor_completemask = tf.logical_or(tf.cast(oormask,dtype = tf.bool), oor_thresholded)
        oor_completemask = tf.cast(oor_completemask,tf.float32)        
        
        depth_masked = self.replace_mask_with_value(depth, oor_completemask, val_2_replace = 0)
        
        oor_completemask = tf.reshape(oor_completemask,[spatial_dim[0], spatial_dim[1], 1])
        depth_unfold = tf.reshape(depth_masked,[-1])
        depth_unfold_orig = tf.cast(depth_unfold,tf.float32)/dstep
        depth_unfold_round = tf.round(depth_unfold_orig)
        delta = depth_unfold_orig - depth_unfold_round
        depth_unfold_round = tf.cast(depth_unfold_round,tf.int64)
                
        good = tf.squeeze(tf.where(tf.logical_and(depth_unfold_round >= 1,depth_unfold_round <= nChannels)))

        depth_unfold_round_filtered = tf.gather(depth_unfold_round,good)

        pix_indices = tf.concat([good,good,good],0)
        depth_bins = tf.concat([depth_unfold_round_filtered - 1,depth_unfold_round_filtered, depth_unfold_round_filtered + 1],0)
        delta_filtered = tf.gather(delta,good)
        
        vals = tf.concat([(0.5 - delta_filtered)/2, 0.5*tf.ones(tf.shape(depth_unfold_round_filtered)), (0.5 + delta_filtered)/2], 0)
        pix_2d = tf.transpose(tf.stack([pix_indices,depth_bins]))
        #dcc = tf.sparse_tensor_to_dense(tf.SparseTensor(tf.transpose(tf.stack([pix_indices,depth_bins])), vals,
        #                               dense_shape = [spatial_dim[0]*spatial_dim[1],nChannels + 2]),validate_indices = False)
        dcc = tf.sparse_to_dense(pix_2d,[spatial_dim[0]*spatial_dim[1],nChannels + 2], vals,validate_indices = False)
        dcc = tf.reshape(dcc, shape = [spatial_dim[0],spatial_dim[1],nChannels+2])
        

        return dcc
    
    def depth_2_discrete(self, depth, dstep, spatial_dim = (113,500)):
        
        depth_quantiz = tf.cast(depth,tf.float32)/dstep
        depth_quantiz = tf.round(depth_quantiz)
        
        depth_quantiz = tf.cast(depth_quantiz, tf.float32)
        return depth_quantiz

    
    def dcc_2_depth(self, dcc, dstep, nChannels, oorchanFlag = True, spatial_dim = (113,500)):
        
        if oorchanFlag:
            dcc, __ = tf.split(dcc,[nChannels + 2, 1], 2)
        
        h,w = spatial_dim
        
        dvals = tf.matmul(tf.ones([h*w, 1],tf.float32), tf.expand_dims(tf.cast(tf.range(0, nChannels + 2)*dstep,tf.float32),0))
        
        
        dcc = tf.reshape(dcc, [h*w, nChannels + 2])
        sumProbs = tf.reduce_sum(dcc,1)
        goodmask = tf.greater(sumProbs,0)
        pixindics = tf.where(goodmask)
        dcc_sel = tf.gather(dcc,pixindics)
        dvals_sel = tf.gather(dvals,pixindics)
        sumProbs_good = tf.gather(sumProbs,pixindics)
        depthval = tf.squeeze(tf.divide(tf.reduce_sum(tf.multiply(dvals_sel,dcc_sel),2),sumProbs_good))
        
        depth = tf.sparse_to_dense(pixindics, [h*w], depthval,validate_indices = False)
        depth = tf.reshape(depth,[h,w])

        #dcc_sparse = tf.sparse_reshape(dcc_sparse,[h*w, nChannels + 2])
        #sumProbs = tf.sparse_reduce_sum(dcc_sparse, 2)
        #goodmask = tf.greater(sumProbs,0)

        return depth            
    

    def replace_mask_with_value(self,data,mask,val_2_replace = 0.0):

        data = tf.cast(tf.squeeze(data),dtype = tf.float32)        
        mask = tf.squeeze(mask)
        mask_indices = tf.where(tf.cast(mask,tf.bool))
        nonmask_indices = tf.where(tf.not_equal(mask,True))

        nonmask_vals = tf.gather_nd(data,nonmask_indices)
        fin_val = tf.ones([tf.shape(mask_indices)[0]],dtype = tf.float32)*val_2_replace

        data_nonmask = tf.sparse_tensor_to_dense(tf.SparseTensor(nonmask_indices,
                                                nonmask_vals,tf.cast(tf.shape(data),dtype = tf.int64)),validate_indices = False)

        data_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(mask_indices,
                                                fin_val,tf.cast(tf.shape(data),dtype = tf.int64)),validate_indices = False)
        fin_data = data_mask + data_nonmask    





        return fin_data

    def preprocess_depth(self, depth, depth_maxrange, oor_maskcorrect = None):

        depth = tf.cast(depth, tf.float32)
        if oor_maskcorrect is not None:
            
            oor_maskcorrect = tf.cast(oor_maskcorrect, tf.bool)  
            maxrange_tensor = tf.constant(depth_maxrange, depth.dtype)
            depth_mask = tf.greater(depth, maxrange_tensor)
            tot_mask = tf.logical_or(oor_maskcorrect, depth_mask)
            
            depth = self.replace_mask_with_value(depth,tot_mask, val_2_replace = depth_maxrange)
        else:
            maxrange_tensor = tf.constant(depth_maxrange, depth.dtype)
            tot_mask = tf.greater(depth, maxrange_tensor)
            depth = self.replace_mask_with_value(depth, tot_mask, val_2_replace = depth_maxrange)    
        
        #depthprocessed = depthprocessed - self.depth_mean
        depthprocessed = depth/depth_maxrange

        depthprocessed = tf.expand_dims(depthprocessed,axis = 2)
        return depthprocessed

    