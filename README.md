# 书生2.5 - 多模态多任务通用大模型
<div align=center>
<img src='./docs/figs/log.png' width=600>
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-lvis-v1-0-minival)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-minival?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-val?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-pascal-voc-2007)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2007?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-pascal-voc-2012)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2012?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-openimages-v6)](https://paperswithcode.com/sota/object-detection-on-openimages-v6?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-crowdhuman-full-body)](https://paperswithcode.com/sota/object-detection-on-crowdhuman-full-body?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/2d-object-detection-on-bdd100k-val)](https://paperswithcode.com/sota/2d-object-detection-on-bdd100k-val?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevformer-v2-adapting-modern-image-backbones/3d-object-detection-on-nuscenes-camera-only)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-camera-only?p=bevformer-v2-adapting-modern-image-backbones)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-inaturalist-2018)](https://paperswithcode.com/sota/image-classification-on-inaturalist-2018?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-places365)](https://paperswithcode.com/sota/image-classification-on-places365?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-places205)](https://paperswithcode.com/sota/image-classification-on-places205?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=internimage-exploring-large-scale-vision)

<!-- 这个代码仓库是InterImage的官方实现。 [InternImage: Exploring Large-Scale Vision Foundation Models with
Deformable Convolutions](https://arxiv.org/abs/2211.05778). -->


[文章](https://arxiv.org/abs/2211.05778) \| [博客](https://zhuanlan.zhihu.com/p/610772005) | [文档](./docs/)



<!-- ## 书生2.5-30亿参数视觉通用主干模型 -->
## 简介
商汤科技与上海人工智能实验室在2023年3月14日联合发布多模态多任务通用大模型“书生2.5”。“书生2.5”在多模态多任务处理能力中斩获多项全新突破，其卓越的图文跨模态任务处理能力可为自动驾驶等通用场景任务提供高效精准的感知和理解能力支持。“书生2.5”致力于多模态多任务通用模型的构建，旨在接收处理各种不同模态的输入，并采用统一的模型架构和参数处理各种不同的任务，促进不同模态和任务之间在表示学习方面的协作，逐步实现通用人工智能领域的融会贯通。

## 概览图

<div align=center>
<img src='./docs/figs/intern_pipeline.png' width=700>
</div>


## 亮点
- :thumbsup: **高达30亿参数的最强视觉通用主干模型**
- 🏆 **斩获ImageNet和COCO“双料冠军”**
- 🏆 **开源模型中ImageNet准确度最高**
- 🏆 **物体检测标杆数据集COCO `65.4 mAP`，唯一超过`65.0 mAP`的模型**

## 最新进展
- 2023年3月14日: 🚀 书生2.5发布！
- 2023年2月28日: 🚀 InternImage 被CVPR 2023接收!
- 2022年11月18日: 🚀 基于 InternImage-XL 主干网络，[BEVFormer v2](https://arxiv.org/abs/2211.10439) 在nuScenes的纯视觉3D检测任务上取得了最佳性能 `63.4 NDS` ！
- 2022年11月10日: 🚀 InternImage-H 在COCO目标检测任务上以 `65.4 mAP` 斩获冠军，是唯一突破 `65.0 mAP` 的超强物体检测模型！
- 2022年11月10日: 🚀 InternImage-H 在ADE20k语义分割数据集上取得 `62.9 mIoU` 的SOTA性能！

## 经典数据集性能
### 1. ImageNet
在图像分类标杆数据集ImageNet上，“书生2.5”仅基于公开数据便达到了 90.1% 的Top-1准确率。这是除谷歌与微软两个未公开模型及额外数据集外，唯一准确率超过90.0%的模型，同时也是世界上开源模型中ImageNet准确度最高，规模最大的模型。

<div align=center>
<img src='./docs/figs/imagenet_leaderboard.png' width=600>
</div>

### 2. COCO
在物体检测标杆数据集COCO上，“书生2.5” 取得了 65.4 的 mAP，是世界上唯一超过65.0mAP的模型，已经领先谷歌和微软。
<div align=center>
<img src='./docs/figs/coco_leaderboard.png' width=600>
</div>

## 图文跨模态应用
图文跨模态是通用人工智能的重要方向，可以结合图像、文本等信息来实现对世界的理解和认知，在自动驾驶等领域具有重要且广泛的应用前景。“书生2.5”多模态多任务通用模型当前专注于图文跨模态任务，模型无需微调，便可灵活、精准地应对各式任务，并在“图文检索、以图生文和以文生图和”等图文跨模态任务上都取得了令人印象深刻的表现。在20余个不同场景不同任务的单模态和跨模态公开数据集， “书生2.5”都取得了最佳成绩。
<div align=center>
<img src='./docs/figs/multi_task_sota.png' width=600>
</div>


### 1. 图文检索

“书生2.5”可根据文本内容需求快速定位检索出语义最相关的图像。这一能力既可应用于视频和图像集合，也可进一步结合物体检测框，具有丰富的应用模式，帮助用户更便捷、快速地找到所需图像资源, 例如可在相册中返回文本所指定的相关图像。

<div align=center>
<img src='./docs/figs/image_text_retrieval.png' width=600>
</div>

### 2. 以图生文

“书生2.5”的“以图生文”在图像描述、视觉问答、视觉推理和文字识别等多个方面均拥有强大的理解能力。例如在自动驾驶场景下，可以提升场景感知理解能力，辅助车辆判断交通信号灯状态、道路标志牌等信息，为车辆的决策规划提供有效的感知信息支持。

<div align=center>
<img src='./docs/figs/img2text.png' width=600>
</div>

### 3. 以文生图
“书生2.5”的“以文生图”能力，可根据用户提出的文本创作需求，利用扩散模型生成算法，生成满足需求的高质量、自然的写实图像。例如，针对自动驾驶等数据依赖场景，“书生2.5”可以生成各类真实的道路交通场景，如繁忙的城市街道、雨天拥挤的车道、马路上奔跑的狗等，从而辅助自动驾驶系统的研发，不断提高Corner Case场景的感知能力上限。
<div align=center>
<img src='./docs/figs/text2img.png' width=600>
</div>



## 核心技术
“书生2.5”在图文跨模态领域卓越的性能表现，源自于在多模态多任务通用模型技术核心方面的多项创新，实现了视觉核心视觉感知大模型主干网络（InternImage）、用于文本核心的超大规模文本预训练网络（LLM）和用于多任务的兼容解码建模（Uni-Perceiver）的创新组合。  视觉主干网络InternImage参数量高达30亿，能够基于动态稀疏卷积算子自适应地调整卷积的位置和组合方式，从而为多功能视觉感知提供强大的表示。Uni-Perceiver通才任务解码建模通过将不同模态的数据编码到统一的表示空间，并将不同任务统一为相同的任务范式，从而能够以相同的任务架构和共享的模型参数同时处理各种模态和任务。


<div align=center>
<img src='./docs/figs/network.png' width=600>
</div>


## 项目功能
- [Coming] InternImage-H(1B)/G(3B)
- [Coming] 各类downstream tasks
- 🚀 TensorRT 推理
- 🚀 InternImage系列分类代码
- 🚀 InternImage-T/S/B/L/XL ImageNet-1k 预训练模型
- 🚀 InternImage-L/XL ImageNet-22k 预训练模型
- 🚀 InternImage-T/S/B/L/XL 检测和实例分割模型
- 🚀 InternImage-T/S/B/L/XL语义分割模型


## 开源模型
- 目标检测和实例分割: [COCO](detection/configs/mask_rcnn/)
- 语义分割: [ADE20K](segmentation/configs/ade20k/), [Cityscapes](segmentation/configs/cityscapes/)



## 经典视觉任务性能

**ImageNet图像分类**
|      name      |   pretrain   | resolution | acc@1 | #param | FLOPs |      22K model      |      1K model       |
| :------------: | :----------: | :--------: | :---: | :-----: | :---: | :-----------------: | :-----------------: |
| InternImage-T  | ImageNet-1K  |  224x224   | 83.5  |   30M   |  5G   |          -          | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_t_1k_224.pth) \| [cfg](classification/configs/internimage_t_1k_224.yaml) |
| InternImage-S  | ImageNet-1K  |  224x224   | 84.2  |   50M   |  8G   |          -          | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_s_1k_224.pth) \| [cfg](classification/configs/internimage_s_1k_224.yaml) |
| InternImage-B  | ImageNet-1K  |  224x224   | 84.9  |   97M   |  16G  |          -          | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_b_1k_224.pth) \| [cfg](classification/configs/internimage_b_1k_224.yaml) |
| InternImage-L  | ImageNet-22K |  384x384   | 87.7  |  223M   | 108G  | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_l_22k_192to384.pth)            | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_l_22kto1k_384.pth) \| [cfg](classification/configs/internimage_l_22kto1k_384.yaml) |
| InternImage-XL | ImageNet-22K |  384x384   | 88.0  |  335M   | 163G  | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_xl_22k_192to384.pth)            | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_xl_22kto1k_384.pth) \| [cfg](classification/configs/internimage_xl_22kto1k_384.yaml) |
| InternImage-H | ImageNet-22K |  224x224   | 88.9  |  1.08B   | 188G  | TBD         | TBD |
| InternImage-H | ImageNet-22K |  640x640   | 89.6  |  1.08B   | 1478G  |TBD           | TBD |
| InternImage-G | ImageNet-22K |  640x640   | 90.1  |  3B   | -  |  TBD      | TBD|

**COCO目标检测和实例分割**

|    backbone    |       method       | schd | box mAP | mask mAP | #param | FLOPs | Download | 
| :------------: | :----------------: | :---------: | :-----: | :------: | :-----: | :---: | :---: | 
| InternImage-T  |     Mask R-CNN     |     1x      |  47.2   |   42.5   |   49M   | 270G  |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_t_fpn_1x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_internimage_t_fpn_1x_coco.py) |
| InternImage-T  |     Mask R-CNN     |     3x      |  49.1   |   43.7   |   49M   | 270G  |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_t_fpn_3x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_internimage_t_fpn_3x_coco.py) |
| InternImage-S  |     Mask R-CNN     |     1x      |  47.8   |   43.3   |   69M   | 340G  |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_s_fpn_1x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_internimage_s_fpn_1x_coco.py) |
| InternImage-S  |     Mask R-CNN     |     3x      |  49.7   |   44.5   |   69M   | 340G  |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_s_fpn_3x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_internimage_s_fpn_3x_coco.py) |
| InternImage-B  |     Mask R-CNN     |     1x      |  48.8   |   44.0   |  115M   | 501G  |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_b_fpn_1x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_internimage_b_fpn_1x_coco.py) |
| InternImage-B  |     Mask R-CNN     |     3x      |  50.3   |   44.8   |  115M   | 501G  |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_b_fpn_3x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_internimage_b_fpn_3x_coco.py) |
| InternImage-L  |     Cascade        |     1x      |  54.9   |   47.7   |  277M   | 1399G |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_l_fpn_1x_coco.pth) \| [cfg](detection/configs/cascade_mask_rcnn/cascade_internimage_l_fpn_1x_coco.py) |
| InternImage-L  |     Cascade        |     3x      |  56.1   |   48.5   |  277M   | 1399G |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_l_fpn_3x_coco.pth) \| [cfg](detection/configs/cascade_mask_rcnn/cascade_internimage_l_fpn_3x_coco.py) |
| InternImage-XL |     Cascade        |     1x      |  55.3   |   48.1   |  387M   | 1782G |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.pth) \| [cfg](detection/configs/cascade_mask_rcnn/cascade_internimage_xl_fpn_1x_coco.py) |
| InternImage-XL |     Cascade        |     3x      |  56.2   |   48.8   |  387M   | 1782G |  [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.pth) \| [cfg](detection/configs/cascade_mask_rcnn/cascade_internimage_xl_fpn_3x_coco.py) |
| InternImage-H |     DINO        |     3x      |  65.0   |   -   |  2.18B   | TBD |  TBD |
| InternImage-G |     DINO        |     3x      |  65.3   |   -   |  3B   | TBD |  TBD |

**ADE20K语义分割**

|    backbone    | resolution | single scale | multi scale | #param | FLOPs | Download | 
| :------------: | :--------: | :----------: | :---------: | :-----: | :---: |   :---:  |
<<<<<<< HEAD
| InternImage-T  |  512x512   |     47.9     |    48.1     |   59M   | 944G  | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_t_512_160k_ade20k.pth) \| [cfg](segmentation/configs/upernet/upernet_internimage_t_512_160k_ade20k.py) |
| InternImage-S  |  512x512   |     50.1     |    50.9     |   80M   | 1017G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_s_512_160k_ade20k.pth) \| [cfg](segmentation/configs/upernet/upernet_internimage_s_512_160k_ade20k.py) |
| InternImage-B  |  512x512   |     50.8     |    51.3     |  128M   | 1185G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_b_512_160k_ade20k.pth) \| [cfg](segmentation/configs/upernet/upernet_internimage_b_512_160k_ade20k.py) |
| InternImage-L  |  640x640   |     53.9     |    54.1     |  256M   | 2526G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_l_640_160k_ade20k.pth) \| [cfg](segmentation/configs/upernet/upernet_internimage_l_640_160k_ade20k.py) |
| InternImage-XL |  640x640   |     55.0     |    55.3     |  368M   | 3142G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_xl_640_160k_ade20k.pth) \| [cfg](segmentation/configs/upernet/upernet_internimage_xl_640_160k_ade20k.py) |
| InternImage-H |  896x896   |     59.9     |    60.3     |  1.12B   | 3566G | TBD |
| InternImage-H |  896x896   |     62.5     |    62.9     |  1.31B   | 4635G | TBD |


**模型推理速度**

|      name      | resolution | #param | FLOPs | Batch 1 FPS(TensorRT) |
| :------------: | :--------: | :-----: | :---: | :-------------------: |
| InternImage-T  |  224x224   |   30M   |  5G   |          156          |
| InternImage-S  |  224x224   |   50M   |  8G   |          129          |
| InternImage-B  |  224x224   |   97M   |  16G  |          116          |
| InternImage-L  |  384x384   |  223M   | 108G  |          56           |
| InternImage-XL |  384x384   |  335M   | 163G  |          47           |


## 引用

若“书生2.5”对您的研究工作有帮助，请参考如下bibtex对我们的工作进行引用。

```
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}

@inproceedings{zhu2022uni,
  title={Uni-perceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks},
  author={Zhu, Xizhou and Zhu, Jinguo and Li, Hao and Wu, Xiaoshi and Li, Hongsheng and Wang, Xiaohua and Dai, Jifeng},
  booktitle={CVPR},
  pages={16804--16815},
  year={2022}
}

@article{zhu2022uni,
  title={Uni-perceiver-moe: Learning sparse generalist models with conditional moes},
  author={Zhu, Jinguo and Zhu, Xizhou and Wang, Wenhai and Wang, Xiaohua and Li, Hongsheng and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2206.04674},
  year={2022}
}

@article{li2022uni,
  title={Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks},
  author={Li, Hao and Zhu, Jinguo and Jiang, Xiaohu and Zhu, Xizhou and Li, Hongsheng and Yuan, Chun and Wang, Xiaohua and Qiao, Yu and Wang, Xiaogang and Wang, Wenhai and others},
  journal={arXiv preprint arXiv:2211.09808},
  year={2022}
}

@article{yang2022bevformer,
  title={BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision},
  author={Yang, Chenyu and Chen, Yuntao and Tian, Hao and Tao, Chenxin and Zhu, Xizhou and Zhang, Zhaoxiang and Huang, Gao and Li, Hongyang and Qiao, Yu and Lu, Lewei and others},
  journal={arXiv preprint arXiv:2211.10439},
  year={2022}
}

@article{su2022towards,
  title={Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information},
  author={Su, Weijie and Zhu, Xizhou and Tao, Chenxin and Lu, Lewei and Li, Bin and Huang, Gao and Qiao, Yu and Wang, Xiaogang and Zhou, Jie and Dai, Jifeng},
  journal={arXiv preprint arXiv:2211.09807},
  year={2022}
}

@inproceedings{li2022bevformer,
  title={Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={ECCV},
  pages={1--18},
  year={2022},
}
```
