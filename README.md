# GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction

## 1. Introduction

![alt text](pics/pipeline.jpg)
A unified LiDAR-visual system achieving geometrically consistent photorealistic rendering and high-granularity surface reconstruction.
We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field. This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction.

Our paper is currently undergoing peer review. The code will be released once the paper is accepted.

[Project page](https://jianhengliu.github.io/Projects/GS-SDF/) | [Paper](https://arxiv.org/pdf/2503.10170) | [Video](https://youtu.be/w_l6goZPfcI)

## 2. Related paper

[GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction](https://arxiv.org/pdf/2503.10170)

[FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/pdf/2408.14035)  

If you use GS-SDF for your academic research, please cite the following paper. 
```bibtex
@article{liu2025gssdflidaraugmentedgaussiansplatting,
      title={GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction}, 
      author={Jianheng Liu and Yunfei Wan and Bowen Wang and Chunran Zheng and Jiarong Lin and Fu Zhang},
      journal={arXiv preprint arXiv:2108.10470},
      year={2025},
}
```