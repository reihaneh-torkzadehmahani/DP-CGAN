# DP-CGAN
This repository contains the implementation of [Differentially Private Conditional GAN(DP-CGAN)](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Torkzadehmahani_DP-CGAN_Differentially_Private_Synthetic_Data_and_Label_Generation_CVPRW_2019_paper.pdf). 
There are two different Conditional GANs(CGAN) that we made them differentially private: [CGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/) and [Advanced CGAN](https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/CGAN.py). CGAN referes to a Vanilla CGAN architecture in which both generator and discriminator consist of two fully connected layers while in Advanced CGAN the generator and discriminator architectures are exactly the same as in [infoGAN](https://arxiv.org/abs/1606.03657).
# Slides
Here is the link to my slides for DP-CGAN paper:

# Reference
If you use this code, please cite the following paper:
```
@inproceedings{torkzadehmahani2019dp,
  title={DP-CGAN: Differentially Private Synthetic Data and Label Generation}, 
  author={Torkzadehmahani, Reihaneh and Kairouz, Peter and Paten, Benedict}, 
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},  
  pages={0--0},
  year={2019}
}
```
