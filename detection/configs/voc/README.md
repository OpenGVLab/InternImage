# PASCAL VOC

## Introduction

PASCAL VOC 2007 is a widely used dataset for object detection, classification, and segmentation tasks in computer vision. Released in 2007, it contains 9,963 images with 24,640 annotated objects across 20 categories, such as people, animals, and vehicles. The dataset is divided into training (2,501 images), validation (2,510 images), and test (4,952 images) sets. Each object is labeled with a class, bounding box, and additional attributes like "difficult" or "truncated." VOC 2007 introduced the mean Average Precision (mAP) metric, which remains a standard for evaluating object detection models.

PASCAL VOC 2012, released in 2012, is an improved version of VOC 2007 with more diverse images and annotations. It contains 11,540 images and 27,450 object instances, covering the same 20 categories. In addition to object detection and classification, VOC 2012 includes more detailed annotations for semantic segmentation. The dataset is split into training (5,717 images), validation (5,823 images), and a test set with hidden labels. VOC 2012 is often used as a benchmark for deep learning models and serves as a foundation for modern object detection and segmentation research.

## Model Zoo

### DINO + CB-InternImage

|     backbone     |  pretrain  | VOC 2007 | VOC 2012 | #param |                           Config                            |                                                       Download                                                       |
| :--------------: | :--------: | :------: | :------: | :----: | :---------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| CB-InternImage-H | Objects365 |   94.0   |   97.2   | 2.18B  | [config](./dino_4scale_cbinternimage_h_objects365_voc07.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_voc0712.pth) |
