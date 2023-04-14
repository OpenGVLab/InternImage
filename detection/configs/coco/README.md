# COCO


## Introduction

Introduced by Lin et al. in [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312v3.pdf)

The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

Splits: The first version of MS COCO dataset was released in 2014. It contains 164K images split into training (83K), validation (41K) and test (41K) sets. In 2015 additional test set of 81K images was released, including all the previous test images and 40K new images.

Based on community feedback, in 2017 the training/validation split was changed from 83K/41K to 118K/5K. The new split uses the same images and annotations. The 2017 test set is a subset of 41K images of the 2015 test set. Additionally, the 2017 release contains a new unannotated dataset of 123K images.


## Model Zoo

### Mask R-CNN + InternImage

|    backbone    |  schd | box mAP | mask mAP | train speed | train time |#param | FLOPs | Config | Download | 
| :------------: |  :---------: | :-----: | :------: | :-----: |:------: | :-----: |:------: | :-----: | :---: |
| InternImage-T  |          1x      |  47.2   |   42.5   | 0.36s / iter |  9h | 49M   | 270G  |  [config](./mask_rcnn_internimage_t_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.log.json) |
| InternImage-T  |          3x      |  49.1   |   43.7   | 0.34s / iter | 26h  |  49M   | 270G  | [config](./mask_rcnn_internimage_t_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.log.json) |
| InternImage-S  |          1x      |  47.8   |   43.3   | 0.40s / iter | 10h  |  69M   | 340G  |  [config](./mask_rcnn_internimage_s_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.log.json) |
| InternImage-S  |          3x      |  49.7   |   44.5   | 0.40s / iter | 30h  |  69M   | 340G  | [config](./mask_rcnn_internimage_s_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.log.json) |
| InternImage-B  |          1x      |  48.8   |   44.0   | 0.45s / iter | 11.5h  |  115M   | 501G  | [config](./mask_rcnn_internimage_b_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_1x_coco.log.json) |
| InternImage-B  |          3x      |  50.3   |   44.8   | 0.45s / iter | 34h  |  115M   | 501G  |  [config](./mask_rcnn_internimage_b_fpn_3x_coco.py)| [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_3x_coco.log.json) |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.

### Cascade Mask R-CNN + InternImage

|    backbone    |         schd | box mAP | mask mAP | train speed |	train time | #param | FLOPs | Config | Download |
| :------------: |  :---------: | :-----: | :------: | :-----: | :---: | :-----: | :---: | :---: | :---: |
| InternImage-L  |        1x      |  54.9   |   47.7   | 0.73s / iter | 18h |  277M   | 1399G | [config](./cascade_internimage_l_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_1x_coco.pth)  |
| InternImage-L  |        3x      |  56.1   |   48.5   | 0.79s / iter | 15h (4n) |  277M   | 1399G | [config](./cascade_internimage_l_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_3x_coco.log.json) |
| InternImage-XL |        1x      |  55.3   |   48.1   | 0.82s / iter | 21h |  387M   | 1782G | [config](./cascade_internimage_xl_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_1x_coco.log.json) |
| InternImage-XL |        3x      |  56.2   |   48.8   | 0.91s / iter | 17h (4n) |  387M   | 1782G | [config](./cascade_internimage_xl_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.log.json) |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.


### DINO + InternImage
|    backbone    |  lr type     | pretrain    |       schd | box mAP | 	train time | #param | Config | Download |
| :------------: |  :---------: |:---------: | :---------: | :-----: |  :---: | :-----: | :---: | :---: | 
| InternImage-T  | layer-wise lr    | ImageNet-1K  |     1x      |  53.9   |  9.5h |  49M    | [config](./dino_4scale_internimage_t_1x_coco_layer_wise_lr.py)     | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_t_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_t_1x_coco.json) |
| InternImage-L  | layer-wise lr    | ImageNet-22K |     1x      |  57.5   |   18h |  241M   |  [config](./dino_4scale_internimage_l_1x_coco_layer_wise_lr.py)    | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_l_1x_coco_layer_wise_lr.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_l_1x_coco_layer_wise_lr.log.json) |
| InternImage-L  | 0.1x backbone lr | ImageNet-22K |     1x      |  57.6   |   18h |  241M   |  [config](./dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.log.json) |

