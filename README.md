### Depth Coefficients for Depth Completion
**Saif Imran** **Yunfei Long** **Xiaoming Liu** **Daniel Morris**

The site contains Tensorflow implementation of our work "Depth Coefficients for Depth Completion" featured in CVPR 2019. More description at the link to the [paper.](https://arxiv.org/abs/1903.05421)

## Overview

The following gives a overall illustration of our work. 
![Image](/images/overview_cropped.png)

## Implementation Details

# Tensorflow installation

# Dataset Generation

# Network
We use the following configuration to train the network. Note in this implementation, we found that by adding a simple auxiliary loss at the encoder network, the performance improves compared to the reported performance in the paper. So we suggest the readers to stick to the new training strategy when training the network. 
![Image](/images/DC_network.png)


# Pretrained Models

# Evaluation

## Video Demo
Here is a video demonstration of the work in a typical KITTI sequence:
![DC_Video](/images/DC.gif)
[Youtube Video](https://www.youtube.com/watch?v=ghDFX2hQbYY)

We will upload the code and the pretrained models soon for the benefit of the community. Stay tuned!
