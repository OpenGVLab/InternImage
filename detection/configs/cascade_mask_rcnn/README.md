# Cascade R-CNN

> [Cascade R-CNN: High Quality Object Detection and Instance Segmentation](https://arxiv.org/abs/1906.09756)

<!-- [ALGORITHM] -->

## Introduction

In object detection, the intersection over union (IoU) threshold is frequently used to define positives/negatives. The threshold used to train a detector defines its quality. While the commonly used threshold of 0.5 leads to noisy (low-quality) detections, detection performance frequently degrades for larger thresholds. This paradox of high-quality detection has two causes: 1) overfitting, due to vanishing positive samples for large thresholds, and 2) inference-time quality mismatch between detector and test hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, composed of a sequence of detectors trained with increasing IoU thresholds, is proposed to address these problems. The detectors are trained sequentially, using the output of a detector as training set for the next. This resampling progressively improves hypotheses quality, guaranteeing a positive training set of equivalent size for all detectors and minimizing overfitting. The same cascade is applied at inference, to eliminate quality mismatches between hypotheses and detectors. An implementation of the Cascade R-CNN without bells or whistles achieves state-of-the-art performance on the COCO dataset, and significantly improves high-quality detection on generic and specific object detection datasets, including VOC, KITTI, CityPerson, and WiderFace. Finally, the Cascade R-CNN is generalized to instance segmentation, with nontrivial improvements over the Mask R-CNN.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143872197-d99b90e4-4f05-4329-80a4-327ac862a051.png"/>
</div>

## Model Zoo

|    backbone    |         schd | box mAP | mask mAP | train speed |	train time | #param | FLOPs | Config | Download | 
| :------------: |  :---------: | :-----: | :------: | :-----: | :---: | :-----: | :---: | :---: | :---: | 
| InternImage-L  |        1x      |  54.9   |   47.7   | 0.73s / iter | 18h |  277M   | 1399G | [config](./cascade_internimage_l_fpn_1x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_l_fpn_1x_coco.pth)  |
| InternImage-L  |        3x      |  56.1   |   48.5   | 0.79s / iter | 15h (4n) |  277M   | 1399G | [config](./cascade_internimage_l_fpn_3x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_l_fpn_3x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_l_fpn_3x_coco.log.json) |
| InternImage-XL |        1x      |  55.3   |   48.1   | 0.82s / iter | 21h |  387M   | 1782G | [config](./cascade_internimage_xl_fpn_1x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.log.json) |
| InternImage-XL |        3x      |  56.2   |   48.8   | 0.91s / iter | 17h (4n) |  387M   | 1782G | [config](./cascade_internimage_xl_fpn_3x_coco.py) | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.pth) \| [log](https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_3x_coco.log.json) |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.

