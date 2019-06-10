from __future__ import print_function

import numpy as np
import sys
import argparse
import os
import tensorflow as tf
import re
import glob
import easydict as edict
import time
from scipy import interpolate
import random
import re, fnmatch

from utils import *

from datasplitter import TrainValDataSplitter
from dataReader import DataReader
from train_resnet_network import ResNetModel, _get_block_sizes
from tensorflow.python import pywrap_tensorflow

LEARNING_RATE = 1e-4*0.5
MOMENTUM = 0.9
AUXLOSSWt = 0.8
MOVINGAVGDECAY = 0.99
DATASET_DIR = '../Data/'
RANDOM_SEED = 666
EPOCHS_TO_TRAIN = 600
WEIGHT_DECAY = 0.0000000001
NUM_EPOCHS_PER_DECAY = 350
REGULARIZER_LAMBDA = 0
UPDATE_MEAN_VAR = True
NUM_STEPS = 125000
train_beta_gamma = True
SNAPSHOT_DIR = './checkpoint_dir/'
MODEL_NAME = 'resnet_ttencoderearlyfusion-nobndepth_decoder_resnet18_varspatialdrop_gaussdc_splitker5_dc80_16Rorigdataorignormaliz_biased_crossentropy_auxlabelsgt2dc_annotated'
LOGDIR = './log_dir/'
SAVE_MODEL_EVERY = 600
SAVE_SUMMARY_EVERY = 20
SUMMARYIMAGE_FLAG = True
EXPOMOVAVGFLAG = False
ptrainmodel_load = False
REGRESSLOSSWt = 10
BATCH_SIZE = 3

KITTI_seqs = ['2011_09_26','2011_09_28', '2011_09_29','2011_09_30','2011_10_03']
KITTI_params = {'crop_size': [352, 420],                  
                  'input_size': [352, 1242],                  
                  'dce_dstep': 100,
                  'dce_nChannels': 80,
                  'depth_nChannels': 1,
                  'depth_maxrange': 8000.0,                  
                  'nSkips_r': 4,
                  'nSkips_c': 0,
                  'cam_set': [2, 3],                  
                  'orig_normalizefac': True,
                  'Uniformflag': False,                                     
                  'Uniform_samp': 500,
                  'Gen_uniformsampflag': False,                  
                  'val_set': 'shortened',
                  'truncated_height_start': 0,
                  'truncated_height_end': 352,
                  'coloraugmentflag' : True}

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
   
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
    
def get_arguments():
    parser = argparse.ArgumentParser(description = "Train_Network")
    parser.add_argument("--dataset_home", type = str, dest = 'dataset_home',default = DATASET_DIR, help = "prepend the directory path to KITTI_Depth folder")
    parser.add_argument("--batch-size", type = int, dest = 'batch_size', default = BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    return parser.parse_args()                                

def save_checkpoint(sess,saver,checkpoint_dir,step = 0, model_name = None):
    """
        To save the checkpoint use to test or pretrain
    """
    if model_name == None:
        model_name = "model.ckpt"       
    

    saver.save(sess,checkpoint_dir + '/' + model_name,global_step = step)
    print('Checkpoint saved successfully ... \n')



params = KITTI_params.copy()
args = get_arguments()

"""
Two datasets are:
1. KITTI
2.Synthia
3. NYU2
"""
datasplitter = TrainValDataSplitter(data_home = args.dataset_home, dataset = 'KITTI', seqs = KITTI_seqs,
                                    split = 0.95,random_seed = RANDOM_SEED,params = params)

learningparams = edict
dataparams = edict
networkparams = edict
modelparams = edict
dccinfo = edict

dccinfo.dce_dstep = params['dce_dstep']
dccinfo.dce_nChannels = params['dce_nChannels']
dccinfo.datasize = params['crop_size']


resnet_size = 18
bottleneck = False
final_size = 2048

DATAFORMAT = 'NHWC'
DEFAULT_DTYPE = tf.float32


learningparams.learning_rate = LEARNING_RATE
learningparams.momentum = MOMENTUM
learningparams.weight_decay = WEIGHT_DECAY
learningparams.regularizer_lambda = REGULARIZER_LAMBDA
learningparams.epochs = EPOCHS_TO_TRAIN
learningparams.numbatches_per_epoch = datasplitter.training_nexamp/BATCH_SIZE
learningparams.numexamp_perepoch = datasplitter.training_nexamp
learningparams.decay_steps = int(learningparams.numbatches_per_epoch * NUM_EPOCHS_PER_DECAY)
learningparams.loss_type = 'biased_crossentropy_auxloss'
learningparams.oorpix_include_inloss = False



dataparams.batch_size = BATCH_SIZE
dataparams.data_format = DATAFORMAT
dataparams.depth_maxrange = params['depth_maxrange']
dataparams.dccinfo = dccinfo


modelparams.checkpoint_dir = SNAPSHOT_DIR
modelparams.model_dir = MODEL_NAME
modelparams.log_dir = LOGDIR

modelparams.global_step = 0  #model_no to start from/load from

#Network-Specific.
#1. ResNet
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

#os.environ["CUDA_VISIBLE_DEVICES"] = ''
tf.set_random_seed(666)
coord = tf.train.Coordinator()


with tf.device('/CPU:0'):
    with tf.name_scope("traininginputs"):

        reader = DataReader(datasplitter.data_home,datasplitter.dataset_name, datasplitter.training_tags,params, 
                    coord,dataset_catgry = datasplitter.dataset_catgry, random_scale = False,random_mirror = True, gt_type = 'annotated', 

                    random_crop = True, data_type = 1, dataoption = 'color_dc_dcclabels', queue_flag = True, num_preprocess_threads = 16)
        traindata_batch, trainlabel_batch = reader.dequeue(BATCH_SIZE)

    with tf.name_scope("validationinputs"):

        reader = DataReader(datasplitter.data_home,datasplitter.dataset_name,datasplitter.validation_tags,params, 
                        coord, dataset_catgry = datasplitter.dataset_catgry,random_scale = False,random_mirror = True, gt_type = 'annotated', 
                        random_crop = True, data_type = 1, dataoption = 'color_dc_dcclabels', queue_flag = True, num_preprocess_threads = 16)
        valdata_batch,vallabel_batch = reader.dequeue(BATCH_SIZE)
    
#dataparams.num_channels = reader.channelinfo.depth_nchannels
dataparams.num_channels = reader.channelinfo.label_nchannels[0]
dataparams.channelinfo = reader.channelinfo

auxlabels_cat = 'gt2dc'

trainlabel_list = split_labelchannels(reader.channelinfo, trainlabel_batch)
vallabel_list = split_labelchannels(reader.channelinfo, vallabel_batch)

resnet_layers = _get_block_sizes(networkparams.resnet_size)
#Define ResNet Model and Prediction for Training
resnet_model = ResNetModel(networkparams = networkparams, dataparams = dataparams, learningparams = learningparams)
nlayers = 20
pred_train, loss_train = resnet_model.build_resnet_colordepth_earlyfusion_splitnobndepth_encoder_decoder_auxdcsplit_wskips_v2upscaling_discreteloss_spatialdropout(traindata_batch,trainlabel_list, 
                                        nlayers, training = True, reuse_flag = False, params = params, gt_auxlabels_cat = auxlabels_cat,
                                        loss_type = learningparams.loss_type)
                                        
l2_losses = [learningparams.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name]


#Define Models and Prediction for Validation

pred_val, loss_val = resnet_model.build_resnet_colordepth_earlyfusion_splitnobndepth_encoder_decoder_auxdcsplit_wskips_v2upscaling_discreteloss_spatialdropout(valdata_batch,
                                                                                    vallabel_list, nlayers, training = False, params = params, gt_auxlabels_cat = auxlabels_cat,
                                                                                    reuse_flag = True, loss_type = learningparams.loss_type)

if learningparams.loss_type in ['biased_crossentropy_auxloss']:
    trainlabel_list[0] = dcc_2_depth(trainlabel_list[0], params['dce_dstep'], params['dce_nChannels'], 
                            batch_size = dataparams.batch_size, spatial_dim = params['crop_size'])
    pred_train,__ = dccpred_maxpeak_depth(pred_train, params['dce_dstep'], params['dce_nChannels'], 
                                       batch_size = dataparams.batch_size, spatial_dim = params['crop_size'])
    #pred_train,__ = dccpred2coeff_maxpeak_depth(pred_train, params['dce_dstep'], params['dce_nChannels'], 
                                        #batch_size = dataparams.batch_size, spatial_dim = params['crop_size'])                                            
        
else:
    trainlabel_list[0] = trainlabel_list[1]
    pred_train = pred_train

if SUMMARYIMAGE_FLAG:
    color_channels = split_datachannels(reader.channelinfo,traindata_batch)
    batch_ids = random.sample(range(0, dataparams.batch_size), 1)
    
    
    
    with tf.name_scope('summary'):        
                        
        
        tf.summary.image("Prediction",pred_train, max_outputs = 1)
        tf.summary.image("Labels", trainlabel_list[0], max_outputs = 1)
        tf.summary.image("Color_Image",color_channels, max_outputs = 1)
            

reg_loss = tf.add_n(l2_losses)

if learningparams.loss_type in ['biased_crossentropy_auxloss']:
    trainmain_loss = tf.reduce_mean(loss_train[0])
    trainaux_loss = tf.reduce_mean(loss_train[1])

    valmain_loss = tf.reduce_mean(loss_val[0])
    valaux_loss =tf.reduce_mean(loss_val[1])
    #+ tf.reduce_mean(traincrossentropy_loss)
    #
    totloss_trainscalar = AUXLOSSWt*trainaux_loss  + trainmain_loss + reg_loss
    totloss_valscalar = AUXLOSSWt*valaux_loss  + valmain_loss + reg_loss

    tf.summary.scalar('Total Training-Loss',totloss_trainscalar)
    tf.summary.scalar('TotalValidation-Loss',totloss_valscalar)
    tf.summary.scalar('Reg-Losses',reg_loss)

    tf.summary.scalar('Train_MainCE-Loss',trainmain_loss)
    tf.summary.scalar('Train_AuxCE-Loss',AUXLOSSWt*trainaux_loss)


    tf.summary.scalar('Val_MainCE-Loss',valmain_loss)
    tf.summary.scalar('Val_AuxCE-Loss',AUXLOSSWt*valaux_loss)

else:

    totloss_trainscalar = tf.reduce_mean(loss_train) + reg_loss
    totloss_valscalar = tf.reduce_mean(loss_val) + reg_loss

    tf.summary.scalar('Training-Loss', totloss_trainscalar)
    tf.summary.scalar('Validation-Loss',totloss_valscalar)
    tf.summary.scalar('Reg-Losses',reg_loss)


train_op = resnet_model.define_trainop_fixedlr_AdamOptimizer(totloss_trainscalar)
restore_var = [var for var in tf.global_variables()]
    
saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 10)

config = tf.ConfigProto(allow_soft_placement = False, log_device_placement = True)
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()




summary_op = tf.summary.merge_all()
"Make Directory"
checkpoint_dir = os.path.join(modelparams.checkpoint_dir, modelparams.model_dir)
logs_dir = os.path.join(modelparams.log_dir, modelparams.model_dir)

if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


with tf.Session(config = config) as sess:


    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    summary_writer.add_graph(sess.graph)

    

    # Saver for storing checkpoints of the model.
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    sess.run(init)    

    if ckpt and ckpt.model_checkpoint_path:
        
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        
        loader = tf.train.Saver(var_list = restore_var)
        load_checkpoint( sess, loader, ckpt.model_checkpoint_path)
        
    else:
        print('No checkpoint file found. Initializing with random-values ... \n')
        
        load_step = modelparams.global_step

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    global_step = tf.train.get_or_create_global_step()
    
    # Iterate over training steps.
    curr_step = load_step

    for epoch in range(int(curr_step/learningparams.numbatches_per_epoch),learningparams.epochs):
        for step in range(int(learningparams.numbatches_per_epoch)):
            start_time = time.time()
            #feed_dict = {step_ph: curr_step}
            
            if curr_step % SAVE_SUMMARY_EVERY == 0:
                
                train_loss, trainmaince, val_loss, valmaince, __, summary = sess.run([totloss_trainscalar,trainmain_loss, totloss_valscalar,valmain_loss, train_op, summary_op])
                
                summary_writer.add_summary(summary,curr_step)
                duration = time.time() - start_time
                print('curr_step {:d},trainloss_mse = {:.7f},trainmaince_loss = {:.7f},valloss_mse = {:.7f}, valmaince_loss = {:.7f},({:.7f} sec/step)'.format(curr_step,train_loss,trainmaince, val_loss,valmaince, duration))
                if curr_step % SAVE_MODEL_EVERY == 0:
                    save_checkpoint(sess,saver, checkpoint_dir,curr_step)
                curr_step = curr_step + 1

            else:
                __ = sess.run(train_op)
                curr_step = curr_step + 1
        print('Epoch no. {:d} completed\n'.format(epoch))

coord.request_stop()
coord.join(threads)
summary_writer.close()
sess.close()






