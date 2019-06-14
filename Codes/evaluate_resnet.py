from __future__ import print_function

import numpy as np
import sys
import argparse
import os
import tensorflow as tf
import re, fnmatch
import glob
import easydict as edict
import time
from scipy import interpolate
import imageio
import scipy.io as io
from PIL import Image

from datasplitter import TrainValDataSplitter
from dataReader import DataReader
from train_resnet_network import ResNetModel, _get_block_sizes
from utils import *



def interpolate_depth(sparse_map,method = 'nearest'):

    xi,yi = np.meshgrid(np.arange(sparse_map.shape[1]),
            np.arange(sparse_map.shape[0]))
    y,x = np.where(np.logical_not(sparse_map == 0))
    z = sparse_map[y,x]
    point = np.vstack((x,y)).T
    interp_map = interpolate.griddata(point,z,(xi,yi),method = method)
    interp_map_nearest = interpolate.griddata(point,z,(xi,yi),method = 'nearest')
    interp_map_new = interp_map.copy()
    interp_map_new[np.isnan(interp_map)] = interp_map_nearest[np.isnan(interp_map)]
    
    return interp_map_new

    
def split_datachannels(channelinfo,data):    
    
    data_stackchannels = tf.split(data,[channelinfo.color_nchannels,channelinfo.depth_nchannels],-1)

    return data_stackchannels

def mae_error(label, pred, depth_maxrange, oormask = []):
    pred[pred > depth_maxrange] = depth_maxrange
    pred[pred < 0] = 0
    
    if len(oormask):
        mask = np.logical_or(np.logical_or(label == 0, label > depth_maxrange), oormask)
    
    else:
        mask = np.logical_or(label == 0, label > depth_maxrange)
    pix_valid = np.logical_not(mask)    
    mae_sumerror = np.nansum(np.abs(pred[pix_valid] - label[pix_valid]))
    mae_error = np.nanmean(np.abs(pred[pix_valid] - label[pix_valid]))
    return mae_sumerror, np.sum(pix_valid), mae_error    
    

#Root-Mean-Square_Error
def rmse_error(label, pred, depth_maxrange, oormask = []):
    pred[pred > depth_maxrange] = depth_maxrange
    pred[pred < 0] = 0   
    if len(oormask):
        mask = np.logical_or(np.logical_or(label == 0, label > depth_maxrange), oormask)        
        
    else:
        mask = np.logical_or(label == 0, label > depth_maxrange)              
    pix_valid = np.logical_not(mask)

    rmse_sumerror = np.nansum((label[pix_valid] - pred[pix_valid]) ** 2)            
    rmse_error = np.sqrt(np.nanmean((label[pix_valid] - pred[pix_valid]) ** 2))
    return rmse_sumerror, np.sum(pix_valid), rmse_error    


def trmse_error(label, pred, depth_maxrange, threshold = 100, oormask = []):
    pred[pred > depth_maxrange] = depth_maxrange
    pred[pred < 0] = 0

    if len(oormask):
        mask = np.logical_or(label == 0, label > depth_maxrange)
        
    else:
        mask = np.logical_or(label == 0, label > depth_maxrange)              
    pix_valid = np.logical_not(mask)

    trmse_sumerror = np.nansum(np.minimum((label[pix_valid] - pred[pix_valid]) ** 2, threshold**2))            
    trmse_error = np.sqrt(np.nanmean(np.minimum((label[pix_valid] - pred[pix_valid]) ** 2,threshold**2)))

    return trmse_sumerror, np.sum(pix_valid), trmse_error

def tmae_error(label, pred, depth_maxrange, threshold = 100, oormask = []):
    pred[pred > depth_maxrange] = depth_maxrange
    pred[pred < 0] = 0

    if len(oormask):
        mask = np.logical_or(np.logical_or(label == 0, label > depth_maxrange), oormask)        
        
    else:
        mask = np.logical_or(label == 0, label > depth_maxrange)               
    
    pix_valid = np.logical_not(mask)

    tmae_sumerror = np.nansum(np.minimum(np.abs(pred[pix_valid] - label[pix_valid]), threshold))
    tmae_error = np.nanmean(np.minimum(np.abs(pred[pix_valid] - label[pix_valid]),threshold))

    return tmae_sumerror, np.sum(pix_valid), tmae_error


LEARNING_RATE = 1e-3
DATASET_DIR = '../Data/'
MOMENTUM = 0.9
MOVINGAVGDECAY = 0.999
RANDOM_SEED = 666
EPOCHS_TO_TRAIN = 600
NUM_EPOCHS_PER_DECAY = 350
WEIGHT_DECAY = 0.0001
REGULARIZER_LAMBDA = 0
UPDATE_MEAN_VAR = True
NUM_STEPS = 125000
train_beta_gamma = True
SNAPSHOT_DIR = './checkpoint_dir/'              
MODEL_NAME = 'resnet_encoderearlyfusion-nobndepth_decoder_resnet18_varspatialdrop_3hotdcproper_splitker5_dc80_16Rorigdataorignormaliz_biased_crossentropy_auxlabelsgt2dc_annotated'
LOGDIR = './log_dir/'
DATAFORMAT = 'NHWC'
DEFAULT_DTYPE = tf.float32

def get_arguments():
    parser = argparse.ArgumentParser(description = "Evaluation" )
    
    parser.add_argument("--sel_image", type = int, dest = 'sel_image', default = None,
                        help= "Image id to pick and save in Test_Directory")
    parser.add_argument("--dataset_home", type = str, dest = 'dataset_home', 
                        default = DATASET_DIR,help = "prepend the directory path to KITTI_Depth folder")                        
    parser.add_argument("--splitFlag", type = str, dest = 'splitFlag', default = 'T', help = "Flag for splitting image to tiles")
    parser.add_argument("--coeftype", type = str, dest = 'coeftype', default = '3coef', help = "depth reconstruction method, methods are 3coef or allcoef")                                                                                                
    
    return parser.parse_args()

def unprocess_depth(depth_data,depth_max = 6000):

    depth_unprocessed = depth_data*depth_max

    return depth_unprocessed    

def load_checkpoint(sess,saver,checkpoint_file):
    """
        To load the checkpoint use to test or pretrain
    """
    print("\nReading Checkpoints.....\n\n")
    
    if glob.glob(checkpoint_file + '*'):
        
        
        saver.restore(sess,checkpoint_file)
        print("\n Checkpoint Loading Success! %s\n\n"% checkpoint_file)
    else:
        print("\n! Checkpoint Loading Failed \n\n")


KITTI_seqs = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']


KITTI_params = {'crop_size': [352, 420],                  
                  'input_size': [352, 1242],                  
                  'dce_dstep': 100,
                  'dce_nChannels': 80,
                  'depth_nChannels': 1,
                  'depth_maxrange': 8000.0, 
                  'Gen_uniformsampflag': False,                                   
                  'nSkips_r': 4,
                  'nSkips_c': 0,
                  'cam_set': [2,3],
                  'Uniformflag': False,
                  'orig_normalizefac': True,                  
                  'val_set': 'shortened',
                  'threshold': 100, 
                  'truncated_height_start': 0,
                  'truncated_height_end': 352,
                  'coloraugmentflag' : False}

params = KITTI_params.copy()

args = get_arguments()

if args.sel_image:
    sel_imageid = args.sel_image
else:
    sel_imageid = []

coeftype = args.coeftype
BATCH_SIZE = 1
#os.environ["CUDA_VISIBLE_DEVICES"] = ''
tf.set_random_seed(RANDOM_SEED)
coord = tf.train.Coordinator()

"""
dataset: "KITTI"
        "NYU2"
        "Synthia"
"""
datasplitter = TrainValDataSplitter(data_home = args.dataset_home, dataset = 'KITTI', seqs = KITTI_seqs,
                                    split = 0.95,random_seed = None, params = params)

learningparams = edict
dataparams = edict
modelparams = edict
networkparams = edict
dccinfo = edict

dccinfo.dce_dstep = params['dce_dstep']
dccinfo.dce_nChannels = params['dce_nChannels']
dccinfo.datasize = params['crop_size']

learningparams.learning_rate = LEARNING_RATE
learningparams.momentum = MOMENTUM
learningparams.weight_decay = WEIGHT_DECAY
learningparams.regularizer_lambda = REGULARIZER_LAMBDA
learningparams.epochs = EPOCHS_TO_TRAIN
learningparams.numbatches_per_epoch = datasplitter.training_nexamp/BATCH_SIZE
learningparams.numexamp_perepoch = datasplitter.training_nexamp
learningparams.decay_steps = int(learningparams.numbatches_per_epoch * NUM_EPOCHS_PER_DECAY)
learningparams.loss_type = 'None'
learningparams.oorpix_include_inloss = True

dataparams.data_format = DATAFORMAT
dataparams.epochs = EPOCHS_TO_TRAIN
dataparams.trainingbatch_count = datasplitter.training_nexamp/BATCH_SIZE 

dataparams.data_format = DATAFORMAT

dataparams.dccinfo = dccinfo

modelparams.checkpoint_dir = SNAPSHOT_DIR
modelparams.model_dir = MODEL_NAME
modelparams.log_dir = LOGDIR

modelparams.global_step = 0  #model_no to start from/load from

#Network-Specific.
#1. ResNet
resnet_size = 18
bottleneck = False
final_size = 2048


networkparams.resnet_size = resnet_size
networkparams.block_sizes = _get_block_sizes(networkparams.resnet_size)
networkparams.block_strides = [1, 2, 2, 2]
networkparams.bottleneck = bottleneck
networkparams.num_filters = 64
networkparams.kernel_size = 7
networkparams.conv_stride = 2
networkparams.first_pool_size = 3
networkparams.first_pool_stride = 2
networkparams.final_size = final_size
networkparams.resnet_version = 1
networkparams.dtype = DEFAULT_DTYPE

dataoption = 'color_dc_dcclabels'

if args.splitFlag == 'T':
    randomCropFlag = False
else:
    randomCropFlag = True

with tf.device('/CPU:0'):
    with tf.name_scope("testinputs"):

        reader = DataReader(datasplitter.data_home, datasplitter.dataset_name, datasplitter.test_tags, params, 
                    coord, dataset_catgry = datasplitter.dataset_catgry, random_scale = False, random_mirror = False, gt_type = 'annotated', 
                    random_crop = randomCropFlag, data_type = 1, dataoption = dataoption, queue_flag = False)
        

dataparams.num_channels = reader.channelinfo.depth_nchannels    
dataparams.channelinfo = reader.channelinfo

queue_list = tf.placeholder(tf.string,[3])

testdata_batch, testlabel_batch = reader.read_data_from_disk(queue_list)
BATCH_SIZE = 1

if args.splitFlag == 'T':
    BATCH_SIZE = 1

    testdata_batch = tf.stack(testdata_batch, 0)
    testlabel_batch = tf.stack(testlabel_batch, 0)


dataparams.batch_size = BATCH_SIZE    


if len(testdata_batch.shape) < 4:
    testdata_batch = tf.expand_dims(testdata_batch,0)
    testlabel_batch = tf.expand_dims(testlabel_batch,0)

data_stack = split_datachannels(reader.channelinfo, testdata_batch)
depth_data = data_stack[1]

label_list = split_labelchannels(reader.channelinfo, testlabel_batch)

resnet_model = ResNetModel(networkparams = networkparams, learningparams = learningparams, dataparams = dataparams)
nlayers = 20

pred_test, __ = resnet_model.build_resnet_colordepth_earlyfusion_splitnobndepth_encoder_decoder_auxdcsplit_wskips_v2upscaling_discreteloss_spatialdropout(testdata_batch,
                            label_list[0], nlayers = nlayers, training = False, reuse_flag = False, loss_type = None, params = params)

if args.splitFlag == 'T':
    ntiles = np.uint8(np.ceil(params['input_size'][1]/params['crop_size'][1]))
    pred_full = tf.zeros([params['input_size'][0]*params['input_size'][1], params['dce_nChannels'] + 2],tf.float32)
    crop_w = params['crop_size'][1]
    act_width = params['input_size'][1]
    
    shape = [params['input_size'][0]*params['input_size'][1], params['dce_nChannels'] + 2]
    
    pred_test = tf.split(pred_test, ntiles, 0)
    for indx in range(ntiles):
        offset_width = indx*crop_w
        if (offset_width + crop_w - 1) > act_width:
            offset_width = act_width - crop_w
        x = np.arange(params['crop_size'][1])
        y = np.arange(params['crop_size'][0])    
        X, Y = np.meshgrid(x, y)    
        
        indices = params['input_size'][1]*Y.flatten() + (X.flatten() + offset_width)
        updates = tf.reshape(pred_test[indx], [params['crop_size'][0]*params['crop_size'][1], params['dce_nChannels'] + 2])    
        pred_full = pred_full + tf.scatter_nd(indices[:, np.newaxis], updates, shape)

    pred_full = tf.nn.softmax(pred_full, 1)
    sum_wt = tf.reduce_sum(pred_full, 1, keepdims = True)
    goodmask = tf.greater(sum_wt,0)
    pred_full = pred_full/sum_wt
    

    pred_test = tf.reshape(pred_full, [1, params['input_size'][0], params['input_size'][1], params['dce_nChannels'] + 2 ])
    pred_dc = pred_test
    if coeftype == '3coef':
        pred_test,__ = dccpred_maxpeak_depth(pred_test, params['dce_dstep'], np.int64(params['depth_maxrange']/params['dce_dstep']), dataparams.batch_size, spatial_dim = params['input_size'], softmaxFlag = False) 
    elif coeftype == 'allcoef':
        pred_test = tf.reshape(pred_test, [-1, params['dce_nChannels'] + 2])
        pred_test = dcc_2_depth(pred_test, params['dce_dstep'], params['dce_nChannels'],dataparams.batch_size, spatial_dim = params['input_size'])
    else:
        ValueError('Depth Reconstruction Method undefined. Exiting ...\n')        
else:
    if coeftype == '3coef':
        pred_test,__ = dccpred_maxpeak_depth(pred_test, params['dce_dstep'],np.int64(params['depth_maxrange']/params['dce_dstep']), dataparams.batch_size, spatial_dim = params['crop_size']) 

    elif coeftype == 'allcoef':
        pred_test = tf.reshape(pred_test, [-1, params['dce_nChannels'] + 2])
        pred_test = tf.nn.softmax(pred_test,1)    
        pred_test = dcc_2_depth(pred_test, params['dce_dstep'], params['dce_nChannels'],dataparams.batch_size, spatial_dim = params['crop_size'])

    else:
        ValueError('Depth Reconstruction Method undefined. Exiting ...\n')                
    
    
        
    
gt_depth = label_list[1]/params['depth_maxrange']    
pred_test = pred_test/params['depth_maxrange']



restore_var = [v for v in tf.global_variables()]


checkpoint_dir = os.path.join(modelparams.checkpoint_dir, modelparams.model_dir)

config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
init = tf.global_variables_initializer()

pix_sumcount = 0
pix_tsumcount = 0
mae_accum = 0
rmse_accum = 0
tmae_accum = 0
trmse_accum = 0

with tf.Session(config = config) as sess:


    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:               
        
        
        loader = tf.train.Saver(var_list = restore_var)
        
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load_checkpoint( sess,loader, ckpt.model_checkpoint_path)
        #loader.restore(sess, checkpoint_dir + '/' + 'model.ckpt-132000')
    else:
        print('No checkpoint file found. Initializing with random-values ... \n')
        sess.run(init)

    testimg_dir = os.path.join('TestSet_Results', modelparams.model_dir, 'model_%d'%load_step)

    if not os.path.exists(testimg_dir):
       os.makedirs(testimg_dir)

    queue_extract = np.transpose(reader.queue.eval())
    if sel_imageid:
        queue_extract = np.array([queue_extract[sel_imageid]])
    
            
    
    tot_testsamples = len(queue_extract)

    for idx, queue in enumerate(queue_extract):
        pred_map, label_map, depth_map = sess.run([pred_test, gt_depth, depth_data], feed_dict = {queue_list:queue})

        if args.splitFlag == 'T':
            label_map = np.array(Image.open(queue[1]))
            label_map = np.float32(label_map)*100/256
            label_map[label_map >= params['depth_maxrange']] = params['depth_maxrange']
            label_map = label_map/params['depth_maxrange']
            label_map = label_map[np.newaxis,:,:, np.newaxis]
        
        color_map = np.array(Image.open(queue[0]))        
        
        pred_map = unprocess_depth(pred_map, depth_max = params['depth_maxrange'])
        label_map = unprocess_depth(label_map,depth_max = params['depth_maxrange'])

        mae_sumscore, pix_count, mae_sing_error = mae_error(label_map, pred_map, params['depth_maxrange'])        
        
        pix_sumcount += pix_count
        print('MAE Error %f cm for test_data %d\n'%(mae_sing_error,idx))
        rmse_sumscore, __, rmse_sing_error = rmse_error(label_map, pred_map, params['depth_maxrange'])
        tmae_sumscore, pix_tcount, tmae_singerror = tmae_error(label_map, pred_map, params['depth_maxrange'],params['threshold'])
        trmse_sumscore, __, trmse_singerror = trmse_error(label_map, pred_map, params['depth_maxrange'],params['threshold'])
        
        mae_accum += mae_sing_error
        rmse_accum += rmse_sing_error
        tmae_accum += tmae_singerror
        trmse_accum += trmse_singerror               

        subsample_map = Image.open(queue[2])
        subsample_map = np.array(subsample_map)
        subsample_map = subsample_map[params['truncated_height_start']:params['truncated_height_end'],:]
    
    mae_accum = mae_accum/tot_testsamples
    rmse_accum = rmse_accum/tot_testsamples
    tmae_accum = tmae_accum/tot_testsamples
    trmse_accum = trmse_accum/tot_testsamples

    print('Final MAE Score is %f\n',mae_accum)
    print('Final RMSE Score is %f\n',rmse_accum)    
    print('Final TMAE Score is %f\n', tmae_accum)
    print('Final TRMSE Score is %f\n', trmse_accum)
    
    

    
