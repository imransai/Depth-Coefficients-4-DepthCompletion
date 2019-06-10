import sys
import argparse
import os

import tensorflow as tf


import re
import glob
import easydict as edict
import time
import numpy as np
from scipy import interpolate
import scipy.io as io

from skimage import filters
from skimage.morphology import erosion, dilation, disk, closing
from skimage.color import rgb2gray



def split_datachannels(channelinfo,data):

    color_channels, __ = tf.split(data,[channelinfo.color_nchannels, channelinfo.data_nchannels - channelinfo.color_nchannels],-1)

    return color_channels



def replace_mask_with_value(data,mask,val_2_replace = 0):

        data = tf.cast(data,dtype = tf.float32)        
        
        mask_indices = tf.where(tf.cast(mask,tf.bool))
        nonmask_indices = tf.where(tf.not_equal(mask,True))

        nonmask_vals = tf.gather_nd(data,nonmask_indices)
        val_2_replace = tf.convert_to_tensor(val_2_replace,dtype = data.dtype)

        
        fin_val = tf.multiply(tf.ones([tf.shape(mask_indices)[0]],dtype = data.dtype),val_2_replace)
        

        data_nonmask = tf.sparse_tensor_to_dense(tf.SparseTensor(nonmask_indices,
                                                nonmask_vals,tf.cast(tf.shape(data),dtype = tf.int64)),validate_indices = False)

        data_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(mask_indices,
                                                fin_val,tf.cast(tf.shape(data),dtype = tf.int64)),validate_indices = False)
        fin_data = data_mask + data_nonmask

        return fin_data

def depth_2_dcc_channelsgeneralize(depth, dstep, depth_maxrange, data_format = 'NHWC', batch_size = 1, spatial_dim = (113,500)):

        
    depth = tf.cast(tf.squeeze(depth),tf.float32)       
    
    nChannels = np.int64(depth_maxrange/dstep)

    oor_completemask = tf.squeeze(depth) > depth_maxrange
    oor_completemask = tf.cast(oor_completemask,tf.float32)        
    
    depth_masked = replace_mask_with_value(depth,oor_completemask, val_2_replace = depth_maxrange)
    
    if data_format == 'NCHW':

        oor_completemask = tf.reshape(oor_completemask, [batch_size, 1, spatial_dim[0], spatial_dim[1]])
    else:
        oor_completemask = tf.reshape(oor_completemask, [batch_size, spatial_dim[0], spatial_dim[1], 1])
    
    depth_unfold = tf.reshape(depth_masked,[-1])
    depth_unfold_orig = tf.cast(depth_unfold,tf.float32)/dstep
    depth_unfold_round = tf.round(depth_unfold_orig)
    delta = depth_unfold_orig - depth_unfold_round
    depth_unfold_round = tf.cast(depth_unfold_round,tf.int64)
            
    good = tf.squeeze(tf.where(tf.logical_and(depth_unfold_round >= 1, depth_unfold_round <= nChannels)))

    depth_unfold_round_filtered = tf.gather(depth_unfold_round,good)

    pix_indices = tf.concat([good,good,good],0)
    depth_bins = tf.concat([depth_unfold_round_filtered - 1,depth_unfold_round_filtered, depth_unfold_round_filtered + 1],0)
    delta_filtered = tf.gather(delta,good)
    
    vals = tf.concat([(0.5 - delta_filtered)/2, 0.5*tf.ones(tf.shape(depth_unfold_round_filtered)), (0.5 + delta_filtered)/2], 0)
    pix_2d = tf.transpose(tf.stack([pix_indices,depth_bins]))        
    
    dcc = tf.sparse_to_dense(pix_2d,[batch_size*spatial_dim[0]*spatial_dim[1], nChannels + 2], vals, validate_indices = False)
    dcc = tf.reshape(dcc, shape = [batch_size,spatial_dim[0],spatial_dim[1], nChannels + 2])
    
    return dcc

def depth_2_dcc(depth, dstep, nChannels, data_format = 'NHWC', batch_size = 1, spatial_dim = (113,500)):

        
        depth = tf.cast(tf.squeeze(depth),tf.float32)       
                
        oor_completemask = tf.squeeze(depth)/dstep > nChannels
        oor_completemask = tf.cast(oor_completemask,tf.float32)        
        
        depth_masked = replace_mask_with_value(depth,oor_completemask, val_2_replace = 0)
        
        if data_format == 'NCHW':

            oor_completemask = tf.reshape(oor_completemask, [batch_size, 1, spatial_dim[0], spatial_dim[1]])
        else:
            oor_completemask = tf.reshape(oor_completemask, [batch_size, spatial_dim[0], spatial_dim[1], 1])
        
        depth_unfold = tf.reshape(depth_masked,[-1])
        depth_unfold_orig = tf.cast(depth_unfold,tf.float32)/dstep
        depth_unfold_round = tf.round(depth_unfold_orig)
        delta = depth_unfold_orig - depth_unfold_round
        depth_unfold_round = tf.cast(depth_unfold_round,tf.int64)
                
        good = tf.squeeze(tf.where(tf.logical_and(depth_unfold_round >= 1, depth_unfold_round <= nChannels)))

        depth_unfold_round_filtered = tf.gather(depth_unfold_round,good)

        pix_indices = tf.concat([good,good,good],0)
        depth_bins = tf.concat([depth_unfold_round_filtered - 1,depth_unfold_round_filtered, depth_unfold_round_filtered + 1],0)
        delta_filtered = tf.gather(delta,good)
        
        vals = tf.concat([(0.5 - delta_filtered)/2, 0.5*tf.ones(tf.shape(depth_unfold_round_filtered)), (0.5 + delta_filtered)/2], 0)
        pix_2d = tf.transpose(tf.stack([pix_indices,depth_bins]))

        if data_format == 'NCHW':        
            dcc = tf.sparse_to_dense(pix_2d,[batch_size, nChannels + 2, spatial_dim[0]*spatial_dim[1]], vals, validate_indices = False)
            dcc = tf.reshape(dcc, shape = [batch_size, nChannels + 2,spatial_dim[0],spatial_dim[1]])
        
        else:
            dcc = tf.sparse_to_dense(pix_2d,[batch_size, spatial_dim[0]*spatial_dim[1], nChannels + 2], vals, validate_indices = False)
            dcc = tf.reshape(dcc, shape = [batch_size,spatial_dim[0],spatial_dim[1], nChannels + 2])
        
        return dcc


def dcc_2_depth(dcc, dstep, nChannels, batch_size = 1, spatial_dim = (113,500)):        
        
    h,w = spatial_dim
    
    dvals = tf.matmul(tf.ones([batch_size*h*w, 1],tf.float32), tf.expand_dims(tf.cast(tf.range(0,nChannels + 2)*dstep,tf.float32),0))
    dcc = tf.reshape(dcc,[-1, nChannels + 2])
    sumProbs = tf.reduce_sum(dcc,1)
    goodmask = tf.greater(sumProbs,0)
    pixindics = tf.where(goodmask)
    dcc_sel = tf.gather(dcc,pixindics)
    dvals_sel = tf.gather(dvals,pixindics)
    sumProbs_good = tf.gather(sumProbs,pixindics)
    depthval = tf.squeeze(tf.divide(tf.reduce_sum(tf.multiply(dvals_sel,dcc_sel),2),sumProbs_good))
    
    depth = tf.sparse_to_dense(pixindics,[batch_size*h*w], depthval, validate_indices = False)
    depth = tf.reshape(depth,[batch_size,h,w,1])        

    return depth


def depth_splitsamples(depth, dcc, depth_maxrange, depth_bins = 1000):

    nbins = np.int16(depth_maxrange/depth_bins)
    binsize = []
    sample_bins = []
    depth = tf.reshape(depth, [-1])
    
    for i_bin in range(nbins):

        pixindics = tf.where(tf.logical_and(tf.greater(depth, i_bin*depth_bins), tf.less_equal(depth, (i_bin + 1)*depth_bins)))    
        
        sample_bins.append(tf.random_shuffle(pixindics))
        binsize.append(tf.size(pixindics))

    dccfin = []
    

    for idx, samples in enumerate(sample_bins):

        sample_finbin, __ = tf.split(samples, [binsize[-1], binsize[idx] - binsize[-1]],0)        

        dccfin.append(tf.gather_nd(dcc, sample_finbin))

    return dccfin, sample_bins, binsize

def dccpred_maxpeak_depth(dccpred, dstep, nChannels, batch_size = 1, spatial_dim = (113,500), softmaxFlag = True):

    
    dccpred = tf.reshape(dccpred, [-1, nChannels + 2])
    if softmaxFlag:
        dccpred = tf.nn.softmax(dccpred,1)
    dccpred = tf.stop_gradient(dccpred)
    wvals = tf.reduce_max(dccpred,1)
    dvals = tf.argmax(dccpred, 1)

    goodpixind = tf.where(tf.logical_and(tf.logical_and(tf.greater(wvals,0), tf.greater(dvals,0)), tf.less(dvals, nChannels + 1)))

    
    main_weight = tf.gather_nd(wvals, goodpixind)
    main_depth = tf.gather_nd(dvals, goodpixind)
    

    closer_depth = main_depth - 1
    far_depth = main_depth + 1

    
    closer_weight = tf.gather_nd(dccpred, tf.concat([goodpixind, tf.expand_dims(closer_depth, 1)], axis = 1))
    far_weight = tf.gather_nd(dccpred, tf.concat([goodpixind, tf.expand_dims(far_depth, 1)], axis = 1))

    
    depth = (tf.multiply(tf.cast(main_depth,tf.float32), main_weight)*dstep + 
            tf.multiply(tf.cast(closer_depth,tf.float32), closer_weight)*dstep + 
            tf.multiply(tf.cast(far_depth,tf.float32), far_weight)*dstep)/(main_weight + closer_weight + far_weight)
    
    pred_depth = tf.scatter_nd(goodpixind, tf.squeeze(depth), [batch_size*spatial_dim[0]*spatial_dim[1]])
    
    pred_depth = tf.reshape(pred_depth,[batch_size, spatial_dim[0], spatial_dim[1], 1])
    
    return pred_depth, dccpred




def prepare_discrete_depthlabel(depth_discrete, nChannels, axis = 1, one_hot = True):
    with tf.name_scope('label_encode'):
        
        depth_discrete = tf.squeeze(depth_discrete, axis = axis) # reducing the channel dimension.
        if one_hot:
            depth_onehot = tf.one_hot(tf.cast(depth_discrete,tf.int32), depth = nChannels, axis = axis)

    return depth_onehot


def split_labelchannels(channelinfo, label_channels):
    labels_list = tf.split(label_channels, channelinfo.label_nchannels, -1)
    return labels_list


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

