### Depth Coefficients for Depth Completion
**Saif Imran** **Yunfei Long** **Xiaoming Liu** **Daniel Morris**

The site contains Tensorflow implementation of our work "Depth Coefficients for Depth Completion" featured in CVPR 2019. More description at the link to the [paper.](https://arxiv.org/abs/1903.05421)

## Overview

The following gives a overall illustration of our work.  
![Image](/images/overview_cropped.png)

# Implementation Details

# Prerequisities
The codebase was developed and tested in Ubuntu 16.10, Tensorflow 1.10 and CUDA 9.0. Please see the tensorflow [installation] (https://www.tensorflow.org/install/pip) for details. 

# Dataset Generation
We use the [KITTI depth completion dataset] (http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) for training our network. KITTI provides Velodyne HDL-64E as raw lidar scans as input and provide semi-dense annotated ground-truth data for training (see the paper [Sparsity Invariant CNNs] (https://arxiv.org/abs/1708.06500)). But what makes our case interesting is how we subsample the 64R raw lidar scans to make it 32R, 16R lidar-scans respectively. We needed to split the lidar rows based on azimuth angle in lidar space (see our paper), so we required KITTI raw dataset to access the raw lidar scans, skip the desired number of rows and then project the lidar scans in the image plane. We provide a sample matlab code that can do the data-mapping between KITTI's depth completion and raw dataset and generate the subsampled data which is used for training eventually.

# Network
We use the following configuration to train the network. In this implementation, we used a Resnet-18 network since the model can be fit easily in a single GPU. However in the paper, we used Resnet-34, and we found the bigger network to improve performance slightly. Note that in this implementation, we also found that by adding a simple auxiliary loss at the encoder network, the performance improves compared to the reported performance in the paper. So we suggest the readers to stick to the new training strategy when training the network. 
![Image](/images/DC_Network.png)


# Pretrained Models

# Evaluation

## Video Demo
Here is a video demonstration of the work in a typical KITTI sequence:
![DC_Video](/images/DC.gif)
[Youtube Video](https://www.youtube.com/watch?v=ghDFX2hQbYY)

We will upload the code and the pretrained models soon for the benefit of the community. Stay tuned!

## Citations
If you use our method and code in your work, please cite the following:

@inproceedings{ depth-coefficients-for-depth-completion, 
  author = { Saif Imran and Yunfei Long and Xiaoming Liu and Daniel Morris },
  title = { Depth Coefficients for Depth Completion },
  booktitle = { In Proceeding of IEEE Computer Vision and Pattern Recognition },
  address = { Long Beach, CA },
  month = { January },
  year = { 2019 },
}
