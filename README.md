# QAD: Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning 


*International Conference on Computer Vision 2023* <br /> 


<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/Leminhbinh0209/QAD?style=for-the-badge" height="25"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Leminhbinh0209/QAD?style=for-the-badge" height="25"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/Leminhbinh0209/QAD?style=for-the-badge" height="25">

## Overview of our framework

![IMG](asset/main_arch.png)

## Installation
- Ubuntu 18.04.5 LTS
- CUDA 10.2
- Python 3.6.10

## Datasets 
* NeuralTextures [[Dataset]](https://github.com/ondyari/FaceForensics) [[Paper]](https://arxiv.org/abs/1904.12356) 
* DeepFakes [[Dataset]](https://github.com/ondyari/FaceForensics)  [[GitHub]](https://github.com/deepfakes/faceswap)
* Face2Face [[Dataset]](https://github.com/ondyari/FaceForensics) [[Paper]](https://arxiv.org/abs/2007.14808)
* FaceSwap [[Dataset]](https://github.com/ondyari/FaceForensics) [[GitHub]](https://github.com/deepfakes/faceswap)
* FaceShifter [[Dataset]](https://github.com/ondyari/FaceForensics)  [[Paper]](https://arxiv.org/abs/1912.13457) 
* CelebDFv2 [[Dataset]](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html)  [[Paper]](https://arxiv.org/abs/1909.12962) 
* FFIW10K [[Dataset]](https://github.com/tfzhou/FFIW)  [[Paper]](https://arxiv.org/abs/2103.04570) 

Training with ResNet50 
```
python train --config ./configs/hkrawp_resnet.yaml
```
Or with EfficientNet-B1
```
python train --config ./configs/hkrawp_effnet.yaml
```
[Pre-trained weights](https://o365skku-my.sharepoint.com/:f:/g/personal/bmle_o365_skku_edu/EjnUUM_MKwVDmWHHwg9kX0gBRhOSAyvv24M8ljruYjh8zg?e=RtTMAN) of QAD with ResNet50 and EfficientNet-B1 over 7 datasets are published.

#
*Star (⭐) if you find it useful, and consider to cite our work*  
```
@inproceedings{le2023qad,
  title={Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning},
  author={Le, Binh M. and Woo, Simon },
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
