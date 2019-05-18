### Depth Coefficients for Depth Completion
**Saif Imran** **Yunfei Long** **Xiaoming Liu** **Daniel Morris**

The site contains description and implementation of our work "Depth Coefficients for Depth Completion" featured in CVPR 2019. It can estimate dense depth in image plane based on sparse depth measurements using active depth sensors like lidars etc. While linear upsampling is straight forward, it results in artifacts including depth pixels being interpolated in empty space across discontinuities between objects.  Current methods use deep networks to upsample and "complete" the missing depth pixels.  Nevertheless, depth smearing between objects remains a challenge.  We propose a new representation for depth called Depth Coefficients (DC) to address this problem. More description at the link to the [paper.](https://arxiv.org/abs/1903.05421)

# Overview

The following gives a overall illustration of our work. 
![Image](/images/overview_cropped.png)

# Video Demo
Here is a video demonstration of the work in a typical KITTI sequence:
![Depth Coefficients](https://j.gifs.com/wVrP0M.gif)

We will upload the code and the pretrained models soon for the benefit of the community. Stay tuned!
