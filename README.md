# InternImage

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-all-in-one-pre-training-via/object-detection-on-lvis-v1-0-minival)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-minival?p=towards-all-in-one-pre-training-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevformer-v2-adapting-modern-image-backbones/3d-object-detection-on-nuscenes-camera-only)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-camera-only?p=bevformer-v2-adapting-modern-image-backbones)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=internimage-exploring-large-scale-vision)

This repository is an official implementation of the [InternImage: Exploring Large-Scale Vision Foundation Models with
Deformable Convolutions](https://arxiv.org/abs/2211.05778).

By Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, Xiaogang Wang, Yu Qiao

## News

- `Nov 18, 2022`: 🚀 InternImage-XL merged into [BEVFormer v2](https://arxiv.org/abs/2211.10439) achieves stae-of-the-art performance of `63.4 NDS` on nuScenes Camera Only.
- `Nov 10, 2022`: 🚀🚀 InternImage-H achieves a new record `65.4 mAP` on COCO detection test-dev and `62.9 mIoU` on
ADE20K, outperforming previous models by a large margin.

## Coming soon
- [ ] Classification/detection/segmentation code of the InternImage series.
- [ ] InternImage-T/S/B/L/XL ImageNet-1k pretrained model.
- [ ] InternImage-L/XL ImageNet-22k pretrained model.
- [ ] InternImage-T/S/B/L/XL detection and instance segmentation model.
- [ ] InternImage-T/S/B/L/XL semantic segmentation model.

## Introduction

**InternImage**, initially described in [arxiv](https://arxiv.org/abs/2211.05778), can be a general backbone for computer vision.
It takes deformable convolution as the core operator to obtain large effective receptive fields, and introducing adaptive spatial aggregation
to reduces the strict inductive bias. Our model makes it possible to learn more stronger and robust models with large-scale parameters from massive data.

<div align=center>
<img src='./figs/arch.png' width=400>
</div>

## Main Results on ImageNet with Pretrained Models

**ImageNet-1K and ImageNet-22K Pretrained InternImage Models**

| name | pretrain | resolution |acc@1 | #params | FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | 
| InternImage-T | ImageNet-1K | 224x224 | 83.5 | 30M | 5G |
| InternImage-S | ImageNet-1K | 224x224 | 84.2 | 50M | 8G |
| InternImage-B | ImageNet-1K | 224x224 | 84.9 | 97M | 16G |
| InternImage-L | ImageNet-22K | 384x384 | 87.7 | 223M | 108G |
| InternImage-XL | ImageNet-22K | 384x384 | 88.0 | 335M | 163G |

## Main Results on Downstream Tasks

**COCO Object Detection**

| backbone | method | lr schedule | box mAP | mask mAP | #params | FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| InternImage-T | Mask R-CNN | 1x | 47.2 | 42.5 | 49M | 270G |
| InternImage-S | Mask R-CNN | 1x | 47.8 | 43.3 | 69M | 340G |
| InternImage-B | Mask R-CNN | 1x | 48.8 | 44.0 | 115M | 501G |
| InternImage-L | Cascade Mask R-CNN | 1x | 54.9 | 47.7 | 277M | 1399G |
| InternImage-XL | Cascade Mask R-CNN | 1x | 55.3 | 48.0 | 387M | 1782G |

**ADE20K Semantic Segmentation**

| backbone | resolution | single scale | multi scale | #params | FLOPs|
| :---: | :---: | :---: | :---: | :---: | :---: | 
| InternImage-T | 512x512 | 47.9 | 48.1 | 59M | 944G | 
| InternImage-S | 512x512 | 50.1 | 50.9 | 80M | 1017G |
| InternImage-B | 512x512 | 50.8 | 51.3 | 128M | 1185G |
| InternImage-L | 640x640 | 53.9 | 54.1 | 256M | 2526G |
| InternImage-XL | 640x640 | 55.0 | 55.3 | 368M | 3142G |

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```
