# QAD: Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning 


*International Conference on Computer Vision (**ICCV**) 2023*<br />  
[([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf))]<br /> 


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

For training the model with `QAD`, a `JSON` file includes directories of training, validation, and test images, organized as in our example `dataset/neuraltextures/data.json`.

## Training
Training with ResNet50 
```
python train --config ./configs/hkrawp_resnet.yaml
```
Or with EfficientNet-B1
```
python train --config ./configs/hkrawp_effnet.yaml
```
## Pre-trained weights
 [Pre-trained weights](https://o365skku-my.sharepoint.com/:f:/g/personal/bmle_o365_skku_edu/Evvcf0QH-IxJuqwMGjMHotsBukNl_5Bb-0J-xrCM_G6evA?e=oTFF1O) are released. If you cannot access, please create an empty Drive folder and send an email to us; we will then upload into your folder.

#
*Star (⭐) if you find it useful, and consider to cite our work*  
```
@inproceedings{le2023qad,
  title={Quality-agnostic deepfake detection with intra-model collaborative learning},
  author={Le, Binh M and Woo, Simon S},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22378--22389},
  year={2023}
}
```
