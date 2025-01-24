<p>
	<a href="./README.md">[English Version]</a>
</p>

# 书生图像 - 大规模视觉基础模型

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-lvis-v1-0-minival)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-minival?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-val?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-pascal-voc-2012)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2012?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-openimages-v6)](https://paperswithcode.com/sota/object-detection-on-openimages-v6?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/object-detection-on-crowdhuman-full-body)](https://paperswithcode.com/sota/object-detection-on-crowdhuman-full-body?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/2d-object-detection-on-bdd100k-val)](https://paperswithcode.com/sota/2d-object-detection-on-bdd100k-val?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-inaturalist-2018)](https://paperswithcode.com/sota/image-classification-on-inaturalist-2018?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-places365)](https://paperswithcode.com/sota/image-classification-on-places365?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-places205)](https://paperswithcode.com/sota/image-classification-on-places205?p=internimage-exploring-large-scale-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevformer-v2-adapting-modern-image-backbones/3d-object-detection-on-nuscenes-camera-only)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-camera-only?p=bevformer-v2-adapting-modern-image-backbones)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internimage-exploring-large-scale-vision/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=internimage-exploring-large-scale-vision)

这个代码仓库是 [InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions](https://arxiv.org/abs/2211.05778) 的官方实现。

\[[论文](https://arxiv.org/abs/2211.05778)\] \[[知乎专栏](https://zhuanlan.zhihu.com/p/610772005)\]

## 亮点

- :thumbsup: **高达 30 亿参数的最强视觉通用主干模型**
- 🏆 **图像分类标杆数据集 ImageNet `90.1% Top1`准确率，开源模型中准确度最高**
- 🏆 **物体检测标杆数据集 COCO `65.5 mAP`，唯一超过 `65 mAP` 的模型**

## 最新进展

- 2024年1月22日: 🚀 在 InternImage 中支持了 [DCNv4](https://github.com/OpenGVLab/DCNv4)!
- 2023年2月28日: 🚀 InternImage 被 CVPR 2023 接收!
- 2022年11月18日: 🚀 基于 InternImage-XL 主干网络，[BEVFormer v2](https://arxiv.org/abs/2211.10439) 在nuScenes的纯视觉3D检测任务上取得了最佳性能 `63.4 NDS` ！
- 2022年11月10日: 🚀 InternImage-H 在 COCO 目标检测任务上以 `65.4 mAP` 斩获冠军，是唯一突破 `65.0 mAP` 的超强物体检测模型！
- 2022年11月10日: 🚀 InternImage-H 在 ADE20K 语义分割数据集上取得 `62.9 mIoU` 的SOTA性能！

## 项目功能

- [ ] 各类下游任务
- [ ] 支持 [CVPR 2023 Workshop on End-to-End Autonomous Driving](https://opendrivelab.com/e2ead/cvpr23)，[详见](https://github.com/OpenGVLab/InternImage/tree/master/autonomous_driving)
- [x] 支持提取模型中间层特征，[详见](classification/extract_feature.py)
- [x] 支持基于 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 的低成本训练，[详见](https://github.com/OpenGVLab/InternImage/tree/master/classification)
- [x] DCNv3 算子预编译 `.whl` 包，[详见](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)
- [x] InternImage-H(1B)/G(3B)
- [x] 支持分类/检测/分割 TensorRT 推理
- [x] InternImage 系列分类代码
- [x] InternImage-T/S/B/L/XL ImageNet-1K 预训练模型
- [x] InternImage-L/XL ImageNet-22K 预训练模型
- [x] InternImage-T/S/B/L/XL 检测和实例分割模型
- [x] InternImage-T/S/B/L/XL 语义分割模型

## 简介

InternImage 是一个由上海人工智能实验室、清华大学等机构的研究人员提出的基于卷积神经网络（CNN）的视觉基础模型。与基于 Transformer 的网络不同，InternImage 以可变形卷积 DCNv3 作为核心算子，使模型不仅具有检测和分割等下游任务所需的动态有效感受野，而且能够进行自适应的空间聚合。

<div align=center>
<img src='./docs/figs/arch.png' width=400>
</div>

与 InternImage 相关的其他项目还包括：预训练算法 M3I-Pretraining，通用解码器 Uni-Perceiver 系列，以及自动驾驶感知通用编码器 BEVFormer 系列。

<div align=left>
<img src='./docs/figs/intern_pipeline.png' width=900>
</div>

## 性能

- 在图像分类标杆数据集 ImageNet 上，InternImage 仅基于公开数据便达到了 90.1% 的 Top-1 准确率。这是除谷歌与微软两个未公开模型及额外数据集外，唯一准确率超过 90.0% 的模型，同时也是世界上开源模型中 ImageNet 准确度最高，规模最大的模型；
- 在物体检测标杆数据集 COCO 上，InternImage 取得了 65.5 的 mAP，是世界上唯一超过 65 mAP 的模型；
- 在另外 16 个重要的视觉基础数据集（覆盖分类、检测和分割任务）上取得世界最好性能。

**分类任务**

<table border="1" width="90%">
	<tr align="center">
        <th colspan="1"> 图像分类 </th><th colspan="2"> 场景分类 </th><th colspan="1"> 长尾分类 </th>
    </tr>
    <tr align="center">
        <th>ImageNet</th><th>Places365</th><th>Places 205</th><th>iNaturalist 2018</th>
    </tr>
    <tr align="center">
        <th>90.1</th><th>61.2</th><th>71.7</th><th>92.6</th>
    </tr>
</table>

**检测任务**

<table border="1" width="90%">
	<tr align="center">
        <th colspan="4"> 常规物体检测 </th><th colspan="2"> 长尾物体检测 </th><th colspan="2"> 自动驾驶物体检测 </th><th colspan="1"> 密集物体检测 </th>
    </tr>
    <tr align="center">
        <th>COCO</th><th>VOC 2007</th><th>VOC 2012</th><th>OpenImage</th><th>LVIS minival</th><th>LVIS val</th><th>BDD100K</th><th>nuScenes</th><th>CrowdHuman</th>
    </tr>
    <tr align="center">
        <th>65.5</th><th>94.0</th><th>97.2</th><th>74.1</th><th>65.8</th><th>63.2</th><th>38.8</th><th>64.8</th><th>97.2</th>
    </tr>
</table>

**分割任务**

<table border="1" width="90%">
	<tr align="center">
        <th colspan="3">语义分割</th><th colspan="1">街景分割</th><th colspan="1">RGBD分割</th>
    </tr>
    <tr align="center">
        <th>ADE20K</th><th>COCO Stuff-10K</th><th>Pascal Context</th><th>CityScapes</th><th>NYU Depth V2</th>
    </tr>
    <tr align="center">
        <th>62.9</th><th>59.6</th><th>70.3</th><th>87.0</th><th>68.1</th>
    </tr>
</table>

## 已发布模型

<details open>
<summary> 开源视觉预训练模型 </summary>
<br>
<div>

|      name      |   pretrain   | resolution | #param |                                               download                                                |
| :------------: | :----------: | :--------: | :----: | :---------------------------------------------------------------------------------------------------: |
| InternImage-L  | ImageNet-22K |  384x384   |  223M  |   [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth)    |
| InternImage-XL | ImageNet-22K |  384x384   |  335M  |   [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth)   |
| InternImage-H  |  Joint 427M  |  384x384   | 1.08B  |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth)   |
| InternImage-G  |  Joint 427M  |  384x384   |   3B   | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_pretrainto22k_384.pth) |

</div>

</details>

<details open>
<summary> ImageNet-1K 图像分类 </summary>
<br>
<div>

|      name      |   pretrain   | resolution | acc@1 | #param | FLOPs |                                                                              download                                                                               |
| :------------: | :----------: | :--------: | :---: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-T  | ImageNet-1K  |  224x224   | 83.5  |  30M   |  5G   |       [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth) \| [cfg](configs/without_lr_decay/internimage_t_1k_224.yaml)       |
| InternImage-S  | ImageNet-1K  |  224x224   | 84.2  |  50M   |  8G   |       [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth) \| [cfg](configs/without_lr_decay/internimage_s_1k_224.yaml)       |
| InternImage-B  | ImageNet-1K  |  224x224   | 84.9  |  97M   |  16G  |       [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth) \| [cfg](configs/without_lr_decay/internimage_b_1k_224.yaml)       |
| InternImage-L  | ImageNet-22K |  384x384   | 87.7  |  223M  | 108G  |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22kto1k_384.pth) \| [cfg](configs/without_lr_decay/internimage_l_22kto1k_384.yaml)  |
| InternImage-XL | ImageNet-22K |  384x384   | 88.0  |  335M  | 163G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22kto1k_384.pth) \| [cfg](configs/without_lr_decay/internimage_xl_22kto1k_384.yaml) |
| InternImage-H  |  Joint 427M  |  640x640   | 89.6  | 1.08B  | 1478G |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_22kto1k_640.pth) \| [cfg](configs/without_lr_decay/internimage_h_22kto1k_640.yaml)  |
| InternImage-G  |  Joint 427M  |  512x512   | 90.1  |   3B   | 2700G |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_22kto1k_512.pth) \| [cfg](configs/without_lr_decay/internimage_g_22kto1k_512.yaml)  |

</div>

</details>

<details open>
<summary> COCO 目标检测和实例分割 </summary>
<br>
<div>

|    backbone    |   method   | schd | box mAP | mask mAP | #param | FLOPs |                                                                                     download                                                                                      |
| :------------: | :--------: | :--: | :-----: | :------: | :----: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-T  | Mask R-CNN |  1x  |  47.2   |   42.5   |  49M   | 270G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth) \| [cfg](detection/configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py) |
| InternImage-T  | Mask R-CNN |  3x  |  49.1   |   43.7   |  49M   | 270G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.pth) \| [cfg](detection/configs/coco/mask_rcnn_internimage_t_fpn_3x_coco.py) |
| InternImage-S  | Mask R-CNN |  1x  |  47.8   |   43.3   |  69M   | 340G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.pth) \| [cfg](detection/configs/coco/mask_rcnn_internimage_s_fpn_1x_coco.py) |
| InternImage-S  | Mask R-CNN |  3x  |  49.7   |   44.5   |  69M   | 340G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.pth) \| [cfg](detection/configs/coco/mask_rcnn_internimage_s_fpn_3x_coco.py) |
| InternImage-B  | Mask R-CNN |  1x  |  48.8   |   44.0   |  115M  | 501G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_1x_coco.pth) \| [cfg](detection/configs/coco/mask_rcnn_internimage_b_fpn_1x_coco.py) |
| InternImage-B  | Mask R-CNN |  3x  |  50.3   |   44.8   |  115M  | 501G  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_3x_coco.pth) \| [cfg](detection/configs/coco/mask_rcnn_internimage_b_fpn_3x_coco.py) |
| InternImage-L  |  Cascade   |  1x  |  54.9   |   47.7   |  277M  | 1399G |   [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_1x_coco.pth) \| [cfg](detection/configs/coco/cascade_internimage_l_fpn_1x_coco.py)   |
| InternImage-L  |  Cascade   |  3x  |  56.1   |   48.5   |  277M  | 1399G |   [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_3x_coco.pth) \| [cfg](detection/configs/coco/cascade_internimage_l_fpn_3x_coco.py)   |
| InternImage-XL |  Cascade   |  1x  |  55.3   |   48.1   |  387M  | 1782G |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_1x_coco.pth) \| [cfg](detection/configs/coco/cascade_internimage_xl_fpn_1x_coco.py)  |
| InternImage-XL |  Cascade   |  3x  |  56.2   |   48.8   |  387M  | 1782G |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth) \| [cfg](detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py)  |

|   backbone    |   method   | box mAP (val/test) | #param | FLOPs | download |
| :-----------: | :--------: | :----------------: | :----: | :---: | :------: |
| InternImage-H | DINO (TTA) |    65.0 / 65.4     | 2.18B  | TODO  |   TODO   |
| InternImage-G | DINO (TTA) |    65.3 / 65.5     |   3B   | TODO  |   TODO   |

</div>

</details>

<details open>
<summary> ADE20K 语义分割 </summary>
<br>
<div>

|    backbone    |   method    | resolution | mIoU (ss/ms) | #param | FLOPs |                                                                                                        download                                                                                                         |
| :------------: | :---------: | :--------: | :----------: | :----: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-T  |   UperNet   |  512x512   | 47.9 / 48.1  |  59M   | 944G  |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k.py)                |
| InternImage-S  |   UperNet   |  512x512   | 50.1 / 50.9  |  80M   | 1017G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_s_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_s_512_160k_ade20k.py)                |
| InternImage-B  |   UperNet   |  512x512   | 50.8 / 51.3  |  128M  | 1185G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_b_512_160k_ade20k.py)                |
| InternImage-L  |   UperNet   |  640x640   | 53.9 / 54.1  |  256M  | 2526G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_640_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_l_640_160k_ade20k.py)                |
| InternImage-XL |   UperNet   |  640x640   | 55.0 / 55.3  |  368M  | 3142G |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_640_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_xl_640_160k_ade20k.py)               |
| InternImage-H  |   UperNet   |  896x896   | 59.9 / 60.3  | 1.12B  | 3566G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_h_896_160k_ade20k.py)                |
| InternImage-H  | Mask2Former |  896x896   | 62.5 / 62.9  | 1.31B  | 4635G | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth) \| [cfg](segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py) |

</div>

</details>

<details>
<summary> 模型推理速度 </summary>
<br>
<div>

[Export classification model from pytorch to tensorrt](classification/README.md#export)

[Export detection model from pytorch to tensorrt](detection/README.md#export)

[Export segmentation model from pytorch to tensorrt](segmentation/README.md#export)

|      name      | resolution | #param | FLOPs | batch 1 FPS (TensorRT) |
| :------------: | :--------: | :----: | :---: | :--------------------: |
| InternImage-T  |  224x224   |  30M   |  5G   |          156           |
| InternImage-S  |  224x224   |  50M   |  8G   |          129           |
| InternImage-B  |  224x224   |  97M   |  16G  |          116           |
| InternImage-L  |  384x384   |  223M  | 108G  |           56           |
| InternImage-XL |  384x384   |  335M  | 163G  |           47           |

在使用 `mmdeploy` 将 PyTorch 模型转为 TensorRT 之前，请确保您已正确编译 DCNv3 的自定义算子，其安装方式如下：

```shell
export MMDEPLOY_DIR=/the/root/path/of/MMDeploy

# prepare our custom ops, you can find it at InternImage/tensorrt/modulated_deform_conv_v3
cp -r modulated_deform_conv_v3 ${MMDEPLOY_DIR}/csrc/mmdeploy/backend_ops/tensorrt

# build custom ops
cd ${MMDEPLOY_DIR}
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
make -j$(nproc) && make install

# install the mmdeploy after building custom ops
cd ${MMDEPLOY_DIR}
pip install -e .
```

关于 `mmdeploy` 编译自定义算子的更多细节，请参考这份[文档](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/01-how-to-build/linux-x86_64.md)。

</div>

</details>

## 相关项目

### 多模态基础模型

- [Uni-Perceiver](https://github.com/fundamentalvision/Uni-Perceiver): 通用感知任务预训练统一框架, 可直接处理 zero-shot 和 few-shot 任务
- [Uni-Perceiver v2](https://arxiv.org/abs/2211.09808): 用于处理图像/图文任务的通用模型
- [M3I-Pretraining](https://github.com/OpenGVLab/M3I-Pretraining): 基于最大化输入和目标的互信息的单阶段预训练范式
- [InternVL](https://github.com/OpenGVLab/InternVL): 领先的多模态大语言模型，在 OCR、多模态推理和对话等任务中表现卓越

### 自动驾驶

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer): 基于 BEV 的新一代纯视觉环视感知方案
- [BEVFormer v2](https://arxiv.org/abs/2211.10439): 融合 BEV 感知和透视图检测的两阶段检测器

## 算法竞赛

- [2022 Waymo 3D Camera-Only Detection Challenge](https://waymo.com/open/challenges/2022/3d-camera-only-detection/): 基于 InternImage，BEVFormer++ 取得赛道冠军
- [nuScenes 3D detection](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera): BEVFormer v2 在 nuScenes 纯视觉检测任务中取得SOTA性能 (64.8 NDS)
- [CVPR 2023 Workshop End-to-End Autonomous Driving](https://opendrivelab.com/e2ead/cvpr23): InternImage 作为 baseline 支持了比赛 [3D Occupancy Prediction Challenge](https://opendrivelab.com/AD23Challenge.html#Track3) 和 [OpenLane Topology Challenge](https://opendrivelab.com/AD23Challenge.html#Track1)

## 引用

若这个工作对您的研究有帮助，请参考如下 BibTeX 对我们的工作进行引用。

```bibtex
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```
