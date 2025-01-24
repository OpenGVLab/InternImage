# NYU-Depth-V2

<!-- [ALGORITHM] -->

## Introduction

The NYU Depth V2 dataset is a comprehensive collection of indoor scene data captured using a Microsoft Kinect device. It is widely utilized in computer vision research, particularly for tasks such as depth estimation and semantic segmentation.

## Model Zoo

### Mask2Former + InternImage

|   backbone    | resolution | mIoU (ss/ms) | #param | FLOPs |                        Config                        |                                                                                                          Download                                                                                                          |
| :-----------: | :--------: | :----------: | :----: | :---: | :--------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-H |  480x480   | 67.1 / 68.1  | 1.07B  | 867G  | [config](./mask2former_internimage_h_480_40k_nyu.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_480_40k_nyu.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_480_40k_nyu.log.json) |
