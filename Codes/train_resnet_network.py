from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as io
from random import *
import re, fnmatch
from utils import *



MOVING_AVERAGE_DECAY = 0.9997
_BATCH_NORM_DECAY = MOVING_AVERAGE_DECAY
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
DEFAULT_PADDING = 'SAME'
DATAFORMAT = 'NCHW'


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
        resnet_size: The number of convolutional layers needed in the model.
    Returns:
        A list of block sizes to use in building the model.
    Raises:
        KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, choices.keys()))
    raise ValueError(err)

def var_dropoutdepth(sp_depth, max_depth = 8000, dropout_rate = 0.2):
        
        b, h, w = sp_depth.shape.as_list()[:3]
        sp_depth = tf.reshape(sp_depth,[-1])    
        mask_validpts = sp_depth > 0
        
        
        prob_data = tf.random_uniform(tf.shape(mask_validpts))
        prob_mask = prob_data > dropout_rate
        pixindics_sel = tf.where(prob_mask)

        sp_valid = tf.gather(sp_depth, pixindics_sel)

        sp_dropped = tf.scatter_nd(pixindics_sel, sp_valid, [h*w*b, 1])

        sp_dropped = tf.reshape(sp_dropped, [b, h, w, 1])
        
        return sp_dropped

def min_pooling(channels, pool_size, strides, data_format, padding = 'SAME', max_value = 200000, name = None):
    if data_format == 'NCHW':
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'

    channel_mask1 = tf.equal(channels,0)
    channels = replace_mask_with_value(channels,channel_mask1, max_value)
    channels = tf.negative(channels)

    channels = tf.layers.max_pooling2d(channels, pool_size = pool_size, strides = strides, padding = padding, data_format = data_format)   
    
    channels = tf.negative(channels)
    channel_mask2 = tf.equal(channels, max_value) 
    channels = replace_mask_with_value(channels, channel_mask2)    

    return channels        

def gt_downsample(gt_depth, stride = 2, spatial_scale = 16, pooling = 'min'):


    #h,w,__ = gt_depth.shape.as_list()
    for _ in range(0,int(spatial_scale/2), stride):
        if pooling == 'min':
            gt_depth = min_pooling(gt_depth, 2, strides = stride, data_format = 'NHWC', padding = 'SAME')
        else:
            gt_depth = tf.layers.max_pooling2d(gt_depth, pool_size = 2, strides = stride, padding = 'SAME', data_format = 'channels_last')


    return gt_depth



def resize_features(input, size, method = 'nearest'):
        if method == 'bilinear':
            return tf.image.resize_images(input,size = size,method = tf.image.ResizeMethod.BILINEAR,
                                          align_corners = True)
        elif method == 'nearest':
            return tf.image.resize_images(input,size = size,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                          align_corners = True)
        elif method == 'bicubic':
            return tf.image.resize_images(input,size=size,method = tf.image.ResizeMethod.BICUBIC,
                                          align_corners = True)
        else:
            raise ValueError('Method not matching any interpolation method\n')

def batch_norm(inputs, training, data_format, name = None):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis= 3 if data_format == 'NHWC' else 1,
        momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON, center = True,
            scale = True, training = training, fused = True, name = name)




def fixed_padding(inputs, kernel_size, data_format):

    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                    Should be a positive integer
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    

    if data_format == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def fixed_padding_1d(inputs, kernel_size, data_format):

    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, channels, width_in] or
        [batch, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                    Should be a positive integer
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    

    if data_format == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [0, 0]])
    return padded_inputs    



def deconvolve_network_wskips_nn(inputs, skiplayer, shape, filters, kernel_size, data_format, training, reuse_flag = False):

    if data_format == 'NCHW':    
        inputs = tf.transpose(inputs,[0,2,3,1])
        inputs = resize_features(inputs, shape, method = 'nearest')
        inputs = tf.transpose(inputs,[0,3,1,2])
    else:
        inputs = resize_features(inputs, shape, method = 'nearest')    

    with tf.variable_scope('deconv_upsample_%d%d'%(kernel_size,kernel_size), reuse = reuse_flag):
        inputs = conv2d_fixed_padding(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = 1,
                                    data_format = data_format)    
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.concat([inputs, skiplayer], axis = 1 if data_format == 'NCHW' else 3)
        
    

    return inputs

def deconvolve_network_wskips_nn_v2(inputs, skiplayer, shape, filters, kernel_size, data_format, training, reuse_flag = None, biasflag = False):

    if data_format == 'NCHW':    
        inputs = tf.transpose(inputs,[0,2,3,1])
        inputs = resize_features(inputs, shape, method = 'nearest')
        inputs = tf.transpose(inputs,[0,3,1,2])
    else:
        inputs = resize_features(inputs, shape, method = 'nearest')    

    with tf.variable_scope('deconv_upsample_%d%d'%(kernel_size,kernel_size), reuse = reuse_flag):
        inputs = conv2d_fixed_padding(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = 1,
                                    data_format = data_format, biasflag = biasflag)    
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        if skiplayer is not None:
        #if skiplayer == True:

            inputs = tf.concat([inputs, skiplayer], axis = 1 if data_format == 'NCHW' else 3)
        
    

    return inputs

def split_overlappingchannels(channels, depth_nchannels, max_split = 5, overlaps = 2,data_format = 'NHWC'):

    output_channels = []
    if data_format == 'NHWC':
        channel_dim = channels.get_shape()
        channel_len = channel_dim[3]
        
        for idx in range(0, channel_len - max_split + 1, max_split - overlaps):
            output_channels.append(tf.slice(channels, [0, 0, 0, idx], [-1, -1, -1, max_split]))

        output_channels.append(tf.slice(channels, [0, 0, 0, depth_nchannels - max_split -1], [-1, -1, -1, max_split]))

    elif data_format == 'NCHW':
        channel_dim = channels.get_shape()
        channel_len = channel_dim[1]
        
        for idx in range(0, channel_len - max_split + 1, max_split - overlaps):
            output_channels.append(tf.slice(channels, [0, idx, 0, 0], [-1, max_split, -1, -1]))        
        
        output_channels.append(tf.slice(channels, [0, depth_nchannels - max_split - 1, 0, 0], [-1, max_split, -1, -1]))        

    else:
        ValueError('Data-Format Unknown. Returning ... \n')
    
    

    return output_channels


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, biasflag = False, trainable = True, name = None):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    if data_format == 'NCHW':
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'

    return tf.layers.conv2d(
        inputs=inputs, filters = filters, kernel_size = kernel_size, strides = strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias = biasflag,
        kernel_initializer = tf.variance_scaling_initializer(), trainable = trainable,
        data_format = data_format, name = name)


def conv1d_fixed_padding(inputs, filters, kernel_size, strides, data_format, biasflag = False, trainable = True, name = None):
    """Strided 1-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding_1d(inputs, kernel_size, data_format)
    if data_format == 'NCHW':
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'

    return tf.layers.conv1d(
        inputs=inputs, filters = filters, kernel_size = kernel_size, strides = strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias = biasflag,
        kernel_initializer = tf.variance_scaling_initializer(), trainable = trainable,
        data_format = data_format, name = name)



def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format, reuse_flag = False, biasflag = False):
    """A single block for ResNet v1, without a bottleneck.
    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    kernel_size = 3
    pre_stride = strides
    post_stride = 1

    if projection_shortcut is not None:
        with tf.variable_scope('projection_shortcut', reuse = reuse_flag):
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(inputs = shortcut, training = training,
                                data_format = data_format)
    
    
    with tf.variable_scope('conv_preker_%dx%d_s%d'%(kernel_size,kernel_size, pre_stride),reuse = reuse_flag):
        inputs = conv2d_fixed_padding(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides,
                                        data_format=data_format, biasflag = biasflag, trainable = training)
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

    with tf.variable_scope('conv_postker_%dx%d_s%d'%(kernel_size, kernel_size, post_stride), reuse = reuse_flag):
        inputs = conv2d_fixed_padding(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = 1, biasflag = biasflag,
                                    data_format=data_format, trainable = training)
        inputs = batch_norm(inputs, training, data_format)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

    return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, data_format, reuse_flag = None, biasflag = False, kernel = 3):
    """Creates one layer of blocks for the ResNet model.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first convolution of the layer.
        bottleneck: Is the block created a bottleneck block.
        block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
        training: Either True or False, whether we are currently training the
        model. Needed for batch norm.        
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs = inputs, filters = filters_out, kernel_size = 1, strides = strides,
            data_format = data_format, biasflag = biasflag, trainable = training, name = 'conv%dx%d_s%d'%(1,1,strides))

    # Only the first block per block_layer uses projection_shortcut and strides
    
    with tf.variable_scope('block_%d'%0):
        inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format, reuse_flag = reuse_flag, biasflag = biasflag)


    for block_idx in range(1, blocks):
        with tf.variable_scope('block_%d'%block_idx):
            inputs = block_fn(inputs, filters, training, None, 1, data_format, reuse_flag = reuse_flag, biasflag = biasflag)

    return inputs



def split_4_discretelabels(data, channelinfo, dataoption = None):

    
    color_channels, depth_channels = tf.split(data,[channelinfo.color_nchannels, channelinfo.depth_nchannels],-1)

    return color_channels, depth_channels


def split_datachannels(data,channelinfo):

    color_channels, depth_channels,oor_channels = tf.split(data,[channelinfo.color_nchannels, channelinfo.depth_nchannels, 
                                                                channelinfo.oor_nchannels],-1)

    return color_channels,depth_channels,oor_channels


class ResNetModel(object):

    """Base class for building the Resnet Model."""

    def __init__(self, networkparams = None, learningparams = None, dataparams = None):
        """Creates a model for classifying an image.
        Args:
        resnet_size: A single integer for the size of the ResNet model.
        bottleneck: Use regular blocks or bottleneck blocks.
        num_classes: The number of classes used as labels.
        num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
        kernel_size: The kernel size to use for convolution.
        conv_stride: stride size for the initial convolutional layer
        first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
        first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
        block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
        block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
        final_size: The expected size of the model after the second pooling.
        resnet_version: Integer representing which version of the ResNet network
            to use. See README for details. Valid values: [1, 2]
        data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
        dtype: The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.
        Raises:
        ValueError: if invalid version is selected.
        """
        self.resnet_size = networkparams.resnet_size
        self.data_format = dataparams.data_format
        self.resnet_version = networkparams.resnet_version
        self.bottleneck = networkparams.bottleneck
        self.data_format = networkparams.data_format        
        self.num_filters = networkparams.num_filters
        self.kernel_size = networkparams.kernel_size
        self.conv_stride = networkparams.conv_stride
        self.first_pool_size = networkparams.first_pool_size
        self.first_pool_stride = networkparams.first_pool_stride
        self.block_sizes = networkparams.block_sizes
        self.block_strides = networkparams.block_strides
        self.final_size = networkparams.final_size
        self.dtype = networkparams.dtype
        
        self.pre_activation = networkparams.resnet_version == 2
        self.channelinfo = dataparams.channelinfo
        self.learningparams = learningparams
        self.dataparams = dataparams
        self.num_channels = dataparams.num_channels
        
        if not self.data_format:
            self.data_format = ('NCHW' if tf.test.is_built_with_cuda() else 'NHWC')

        
        if self.resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2. See README for citations.')        

        if self.bottleneck:                       
            
            self.block_fn = _bottleneck_block_v1
            
        else:            
            self.block_fn = _building_block_v1                                
            

        if self.dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        

    def __call__(self, inputs, labels, training):
        """Add operations to classify a batch of input images.
        Args:
        inputs: A Tensor representing a batch of input images.
        training: A boolean. Set to True to add operations required only when
            training the classifier.
        Returns:
        A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        with tf.variable_scope('resnet_model'):
            if self.data_format == 'NCHW':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
                labels = tf.transpose(labels,[0, 3, 1, 2])
            inputs = conv2d_fixed_padding(inputs = inputs, filters=self.num_filters, kernel_size = self.kernel_size,
                                        strides = self.conv_stride, data_format = self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')

            # We do not include batch normalization or activation functions in V2
            # for the initial conv1 because the first ResNet unit will perform these
            # for both the shortcut and non-shortcut paths as part of the first
            # block's projection. Cf. Appendix of [2].
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs = inputs, pool_size=self.first_pool_size,
                    strides = self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = block_layer(
                    inputs=inputs, filters = num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=training,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format)

            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'NCHW' else [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.reshape(inputs, [-1, self.final_size])
            inputs = tf.layers.dense(inputs=inputs, units= self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
        return inputs
    
    
    

    def build_resnet_colordepth_earlyfusion_splitnobndepth_encoder_decoder_auxdcsplit_wskips_v2upscaling_discreteloss_spatialdropout(self, inputs, labels, nlayers, training,
                                                                                            oorchannel = None, reuse_flag = False, sample_weight = None,
                                                                                            loss_type = 'validpdf', weight_file = None, params = None,
                                                                                            gt_auxlabels_cat = 'dc2dc',debugResponseFlag = False):

        color_channels, depth_channels = split_4_discretelabels(inputs, channelinfo = self.channelinfo)

        if self.data_format == 'NCHW':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            #colorchannels = tf.transpose(colorchannels, [0, 3, 1, 2])
            depth_channels = tf.transpose(depth_channels,[0, 3, 1, 2])
            
            labels = tf.transpose(labels,[0, 3, 1, 2])
            color_channels = tf.transpose(color_channels,[0, 3, 1, 2])

        depth_input = depth_channels
        color_input = color_channels
        if training:

            depth_orig = dcc_2_depth(depth_input, params['dce_dstep'], np.int64(params['depth_maxrange']/params['dce_dstep']),
                            self.dataparams.batch_size, spatial_dim = params['crop_size'])
            droprate = tf.random_uniform([], maxval = 0.1)
            depth_drop = var_dropoutdepth(depth_orig, max_depth = params['depth_maxrange'], dropout_rate = droprate)
            depth_input = depth_2_dcc_channelsgeneralize(depth_drop, params['dce_dstep'], params['depth_maxrange'],
                                data_format = 'NHWC', batch_size = self.dataparams.batch_size, spatial_dim = params['crop_size'])

        pre_kernel = 3
        pre_stride = 1
        pre_depth_filters = 80
        pre_color_filters = 48

        encode_filters = 512
        

        with tf.variable_scope('pre_depth_layer_%d'%0, reuse = reuse_flag):
            depth_input = conv2d_fixed_padding(inputs = depth_input, filters = pre_depth_filters, kernel_size = pre_kernel,
                                            strides = pre_stride, data_format = self.data_format,
                                            name = 'conv%dx%d_s%d'%(pre_kernel,pre_kernel,pre_stride))            
            depth_input = tf.nn.relu(depth_input)
            

        with tf.variable_scope('pre_color_layer_%d'%0, reuse = reuse_flag):
            color_input = conv2d_fixed_padding(inputs = color_input, filters = pre_color_filters, kernel_size = pre_kernel,
                                            strides = pre_stride, data_format = self.data_format,
                                            name = 'conv%dx%d_s%d'%(pre_kernel,pre_kernel,pre_stride))
            color_input = batch_norm(color_input, training, self.data_format)
            color_input = tf.nn.relu(color_input)

        concat_input = tf.concat([color_input, depth_input],axis = 1 if self.data_format == 'NCHW' else 3)
        skiplayers = []
        shape_list = []

        skiplayers.append(concat_input)
        shape_list.append(concat_input.get_shape()[2:] if self.data_format == 'NCHW' else concat_input.get_shape()[1:3])

        self.num_filters = [128, 128, 256, 512]

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters[i]
            with tf.variable_scope('encodelayer_%d'%i):

                concat_input = block_layer(inputs = concat_input, filters = num_filters, bottleneck = self.bottleneck,
                                    block_fn = self.block_fn, blocks = num_blocks,
                                    strides = self.block_strides[i], training = training, data_format=self.data_format,reuse_flag = reuse_flag)

                skiplayers.append(concat_input)
                shape_list.append(concat_input.get_shape()[2:] if self.data_format == 'NCHW' else concat_input.get_shape()[1:3])

        with tf.variable_scope('encodelayer_%d'%(i+1), reuse = reuse_flag):
            
            final_encoded = conv2d_fixed_padding(inputs = concat_input, filters = encode_filters, kernel_size = 3,
                                            strides = 2, data_format = self.data_format,
                                            name = 'conv%dx%d_s%d'%(3,3,2))
            final_encoded = batch_norm(final_encoded, training, self.data_format)
            deconvolve_output = tf.nn.relu(final_encoded)

        with tf.variable_scope('finlayer_aux', reuse = reuse_flag):
            pred_aux = conv2d_fixed_padding(inputs = deconvolve_output, filters = self.num_channels, kernel_size = 1,
                                        strides = 1, data_format = self.data_format, biasflag = True,
                                        name = 'conv%dx%d_s%d'%(1,1,1))

        deconv_filters = [64, 64, 64, 128, 256]
        deconv_kernelsize = 3

        for deconv_idx in range(len(skiplayers)-1,-1,-1):
            with tf.variable_scope('deconvlayer_%d'%deconv_idx):
                deconvolve_output = deconvolve_network_wskips_nn_v2(deconvolve_output, skiplayers[deconv_idx],
                                    shape_list[deconv_idx], deconv_filters[deconv_idx], deconv_kernelsize,
                                    self.data_format, training, reuse_flag = reuse_flag)
        post_kernel = 1
        post_stride = 1

        with tf.variable_scope('fin_layer_%d'%nlayers, reuse = reuse_flag):
            pred = conv2d_fixed_padding(inputs = deconvolve_output, filters = self.num_channels, kernel_size = 1,
                                        strides = 1, data_format = self.data_format, biasflag = True, name = 'conv%dx%d_s%d'%(post_kernel,post_kernel,post_stride))
        loss = []
        if loss_type == 'biased_crossentropy_auxloss':

            h,w = self.dataparams.dccinfo.datasize

            

            labels_gt_reducd = gt_downsample(labels[1])
            b,h_aux,w_aux = labels_gt_reducd.shape.as_list()[:3]
            labels_dc_reducd = depth_2_dcc_channelsgeneralize(labels_gt_reducd,
                        dstep = params['dce_dstep'], depth_maxrange = params['depth_maxrange'],
                        batch_size = b, spatial_dim = (h_aux,w_aux))
                #labels_dc_reducd = tf.squeeze(gt_dc, 0)                            
            

            dcc = tf.reshape(labels[0], [h*w*self.dataparams.batch_size, self.num_channels])
            raw_prediction = tf.reshape(pred,[-1, self.num_channels])

            dcc_aux = tf.reshape(labels_dc_reducd, [h_aux*w_aux*self.dataparams.batch_size, self.num_channels])
            raw_prediction_aux = tf.reshape(pred_aux,[-1, self.num_channels])

            sumProbs = tf.reduce_sum(dcc,1)
            sumProbs_aux = tf.reduce_sum(dcc_aux, 1)
            goodmask = tf.not_equal(sumProbs,0)
            pixindics = tf.where(goodmask)

            goodmask_aux = tf.not_equal(sumProbs_aux, 0)
            pixindics_aux = tf.where(goodmask_aux)

            gtdcc_sel = tf.gather(dcc, pixindics)
            biased = tf.reduce_sum(gtdcc_sel*tf.log(gtdcc_sel + 1e-7), axis = 2)
            prediction_sel = tf.gather(raw_prediction, pixindics)

            gtdcc_sel = tf.stop_gradient(gtdcc_sel)

            gtdcc_auxsel = tf.gather(dcc_aux, pixindics_aux)
            biased_aux = tf.reduce_sum(gtdcc_auxsel*tf.log(gtdcc_auxsel + 1e-7), axis = 2)
            prediction_auxsel = tf.gather(raw_prediction_aux, pixindics_aux)

            gtdcc_auxsel = tf.stop_gradient(gtdcc_auxsel)
            # Pixel-wise softmax loss.
            loss_main = tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction_sel, labels = gtdcc_sel) + biased
            loss.append(loss_main)
            loss_aux = tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction_auxsel, labels = gtdcc_auxsel) + biased_aux
            loss.append(loss_aux)        
        else:
            loss = None

        return pred, loss
        
    def define_trainop_fixedlr_MomentumOptimizer(self,loss):
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            opt = tf.train.MomentumOptimizer(self.learningparams.learning_rate, self.learningparams.momentum)
            grads = opt.compute_gradients(loss)            
        
        train_op = opt.apply_gradients(grads)
        
        with tf.name_scope('summary'):            

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None and 'batch_normalization' not in var.name:
                    
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    tf.summary.histogram(var.name,var)               

        return train_op
      

    def train_ops_multigpus(self, opt, grads, batchnorm_updates):

        global_step = tf.Variable(0, trainable = False)            
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        
        train_ops = tf.group(opt.apply_gradients(grads, global_step), batchnorm_updates_op)

        return train_ops
       
    
    
    def define_trainop_steplr_AdamOptimizer(self,loss, learningrate):
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.Variable(0, trainable = False)

        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(learningrate)
            grads = opt.compute_gradients(loss)            
        
        
        train_op = opt.apply_gradients(grads, global_step = global_step)

        with tf.name_scope('summary'):            

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None and 'batch_normalization' not in var.name:
                    
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    tf.summary.histogram(var.name,var)                    
        
        return train_op

    def define_trainop_fixedlr_AdamOptimizer(self,loss):
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.Variable(0, trainable = False)

        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(self.learningparams.learning_rate)
            grads = opt.compute_gradients(loss)            
        
        
        train_op = opt.apply_gradients(grads, global_step = global_step)

        with tf.name_scope('summary'):            

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None and 'batch_normalization' not in var.name:
                    
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    tf.summary.histogram(var.name,var)                    
        
        return train_op

    

    