# 3D-GP-LMVIC: Learning-based Multi-View Image Coding with 3D Gaussian Geometric Priors
Yujun Huang, Bin Chen, Baoyi An, Shu-Tao Xia<br>

Abstract: *Multi-view image compression is vital for 3D-related applications. To effectively model correlations between views, existing methods typically predict disparity between two views on a 2D plane, which works well for small disparities, such as in stereo images, but struggles with larger disparities caused by significant view changes. To address this, we propose a novel approach: learning-based multi-view image coding with 3D Gaussian geometric priors (3D-GP-LMVIC). Our method leverages 3D Gaussian Splatting to derive geometric priors of the 3D scene, enabling more accurate disparity estimation across views within the compression model. Additionally, we introduce a depth map compression model to reduce redundancy in geometric information between views. A multi-view sequence ordering method is also proposed to enhance correlations between adjacent views. Experimental results demonstrate that 3D-GP-LMVIC surpasses both traditional and learning-based methods in performance, while maintaining fast encoding and decoding speed.*

## Setup
Install dependencies:
```shell
pip install -r requirements.txt
```
Please also install [`diff-gaussian-rasterization-w-depth`](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth) and [`simple-knn`](https://github.com/dreamgaussian/dreamgaussian). It is recommended to install them in an environment with CUDA 11.x and GCC 9.4.0, as using higher versions of GCC may result in installation failures.
