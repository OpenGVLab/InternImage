# LVIS

## Introduction

LVIS is a dataset for long tail instance segmentation. It has annotations for over 1000 object categories in 164k images.

## Model Zoo

### DINO + CB-InternImage

Here, we report the box AP on the minival set and the val set, respectively.

|     backbone     |  pretrain  | minival (ss) | val (ss/ms) | #param |                                Config                                 |                                                     Download                                                      |
| :--------------: | :--------: | :----------: | :---------: | :----: | :-------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
| CB-InternImage-H | Objects365 |     65.8     | 62.3 / 63.2 | 2.18B  | [config](./dino_4scale_cbinternimage_h_objects365_lvis_minival_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_lvis.pth) |
