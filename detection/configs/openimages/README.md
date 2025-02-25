# OpenImages

## Introduction

OpenImages V6 is a large-scale dataset , consists of 9 million training images, 41,620 validation samples, and 125,456 test samples. It is a partially annotated dataset, with 9,600 trainable classes.

## Model Zoo

### DINO + CB-InternImage

|     backbone     |  pretrain  | mAP (ss) | #param |                               Config                                |                                                        Download                                                         |
| :--------------: | :--------: | :------: | :----: | :-----------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
| CB-InternImage-H | Objects365 |   74.1   | 2.18B  | [config](./dino_4scale_cbinternimage_h_objects365_openimages_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_openimages.pth) |
