# Pascal Context 59

<!-- [ALGORITHM] -->

## Introduction

The PASCAL Context dataset is an extension of the PASCAL VOC 2010 dataset, providing comprehensive pixel-wise annotations for over 400 classes, including the original 20 object categories and additional background classes. Due to the sparsity of many object categories, a subset of the 59 most frequent classes is commonly used for tasks like semantic segmentation.

## Model Zoo

### Mask2Former + InternImage

|   backbone    | resolution | mIoU (ss/ms) | #param | FLOPs |                               Config                               |                                                                                                                        Download                                                                                                                        |
| :-----------: | :--------: | :----------: | :----: | :---: | :----------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-H |  480x480   | 69.7 / 70.3  | 1.07B  | 867G  | [config](./mask2former_internimage_h_480_40k_pascal_context_59.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_480_40k_pascal_context_59.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_480_40k_pascal_context_59.log.json) |
