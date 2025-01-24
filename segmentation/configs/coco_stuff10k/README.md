# COCO-Stuff-10K

<!-- [ALGORITHM] -->

## Introduction

COCO-Stuff-10K is a dataset designed to enhance scene understanding tasks in computer vision by providing pixel-level annotations for both "things" (discrete objects with well-defined shapes, like cars and people) and "stuff" (amorphous background regions, such as grass and sky). This dataset augments 10,000 images from the original COCO dataset, offering detailed labels across 182 classes—91 "thing" classes and 91 "stuff" classes.

## Model Zoo

### Mask2Former + InternImage

|   backbone    | resolution | mIoU (ss/ms) | #param | FLOPs |                                Config                                 |                                                                                                                           Download                                                                                                                           |
| :-----------: | :--------: | :----------: | :----: | :---: | :-------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-H |  512x512   | 59.2 / 59.6  | 1.28B  | 1528G | [config](./mask2former_internimage_h_512_40k_cocostuff164k_to_10k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_512_40k_cocostuff164k_to_10k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_512_40k_cocostuff164k_to_10k.log.json) |
