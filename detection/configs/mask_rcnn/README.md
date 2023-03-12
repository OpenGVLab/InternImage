# Mask R-CNN

> [Mask R-CNN](https://arxiv.org/abs/1703.06870)

<!-- [ALGORITHM] -->

## Introduction

Mask R-CNN is a conceptually simple, flexible, and general framework for object instance segmentation. It efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. And it extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. 

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967081-c2552bed-9af2-46c4-ae44-5b3b74e5679f.png"/>
</div>

## Model Zoo


|    backbone    |  schd | box mAP | mask mAP | train speed | train time |#param | FLOPs | Config | Download | 
| :------------: |  :---------: | :-----: | :------: | :-----: |:------: | :-----: |:------: | :-----: | :---: |
| InternImage-T  |          1x      |  47.2   |   42.5   | 0.36s / iter |  9h | 49M   | 270G  |  [config](./mask_rcnn_internimage_t_fpn_1x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_t_fpn_1x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_t_fpn_1x_coco.log.json) |
| InternImage-T  |          3x      |  49.1   |   43.7   | 0.34s / iter | 26h  |  49M   | 270G  | [config](./mask_rcnn_internimage_t_fpn_3x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_t_fpn_3x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_t_fpn_3x_coco.log.json) |
| InternImage-S  |          1x      |  47.8   |   43.3   | 0.40s / iter | 10h  |  69M   | 340G  |  [config](./mask_rcnn_internimage_s_fpn_1x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_s_fpn_1x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_s_fpn_1x_coco.log.json) |
| InternImage-S  |          3x      |  49.7   |   44.5   | 0.40s / iter | 30h  |  69M   | 340G  | [config](./mask_rcnn_internimage_s_fpn_3x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_s_fpn_3x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_s_fpn_3x_coco.log.json) |
| InternImage-B  |          1x      |  48.8   |   44.0   | 0.45s / iter | 11.5h  |  115M   | 501G  | [config](./mask_rcnn_internimage_b_fpn_1x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_b_fpn_1x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_b_fpn_1x_coco.log.json) |
| InternImage-B  |          3x      |  50.3   |   44.8   | 0.45s / iter | 34h  |  115M   | 501G  |  [config](./mask_rcnn_internimage_b_fpn_3x_coco.py)| [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_b_fpn_3x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/mask_rcnn_internimage_b_fpn_3x_coco.log.json) |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.

