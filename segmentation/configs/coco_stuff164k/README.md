# COCO-Stuff-164K

<!-- [ALGORITHM] -->

## Introduction

The Common Objects in COntext-stuff (COCO-stuff) dataset is a dataset for scene understanding tasks like semantic segmentation, object detection and image captioning. It is constructed by annotating the original COCO dataset, which originally annotated things while neglecting stuff annotations.  There are 164k images in COCO-Stuff-164K dataset that span over 172 categories including 80 things, 91 stuff, and 1 unlabeled class.


## Model Zoo

### Mask2Former + InternImage

| backbone       | resolution | mIoU (ss) | train speed | train time | #param | FLOPs | Config | Download            |
|:--------------:|:----------:|:-----------:|:-----------:|:----------:|:-------:|:-----:|:-----:|:-------------------:|
| InternImage-H  | 896x896    | 52.6  | 1.6s / iter       | 1.5d (2n)       | 1.31B    | 4635G | [config](./mask2former_internimage_h_896_80k_cocostuff164k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff164k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_896_80k_cocostuff164k.log.json) |
