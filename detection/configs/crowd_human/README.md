# CrowdHuman


## Introduction

Introduced by Shao et al. in [CrowdHuman: A Benchmark for Detecting Human in a Crowd](https://arxiv.org/pdf/1805.00123.pdf)

CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

## Prepare the data
Download the original dataset from [CrowdHuman](https://www.crowdhuman.org/download.html). Then convert annotations by detection/tools/create_crowd_anno.py

- Data Tree of CrowdHuman should look like:
  ```bash
  $ tree CrowdHuman
  CrowdHuman
  ├── annotations
  │   ├── annotation_train.json
  │   ├── annotation_train.odgt
  │   ├── annotation_val.json
  │   ├── annotation_val.odgt
  │   └── ...
  └── Images
      ├── 1074488,79b360006b38332b.jpg
      ├── 1074488,79d54000c6f9d9e5.jpg
      └── ...

## Model Zoo


### Cascade Mask R-CNN + InternImage


|    backbone    |         schd | box mAP | mask mAP | train speed | 	train time | #param | FLOPs | Config | Download |
| :------------: |  :---------: |:-------:|:--------:|:-----------:|:-----------:|:------:|:-----:| :---: |:--------:|
| InternImage-XL |        3x      |   TBD   |   TBD    |     TBD     |     TBD     |  TBD   |  TBD  | [config](./cascade_internimage_xl_fpn_3x_crowd_human.py) |   TBD    |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.

