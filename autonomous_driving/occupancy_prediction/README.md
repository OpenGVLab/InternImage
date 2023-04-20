## InternImage-based Baseline for CVPR23 Occupancy Prediction Challenge!!!!

We improve our baseline with a more powerful image backbone: **InaternImage**, which shows its excellent ability within a series of leaderboards and benchmarks, such as *COCO* and *nuScenes*.


#### openmmlab packages requirements
```bash
torch==1.12 # recommend
mmcv-full>=1.5.0
mmdet==2.24.0
mmsegmentation==0.24.0
timm
numpy==1.22
mmdet3d==0.18.1
```

### Install DCNv3 for InternImage
```bash
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
bash make.sh # requires torch>=1.10
```

### Train with InternImage-Small

```bash
./tools/dist_train.sh projects/configs/bevformer/bevformer_intern-s_occ.py 8 # consumes less than 14G memory
```

Notes: InatenImage provides abundant pre-trained model weights that can be used!!!


### Performance compared to baseline

model name|weight| mIoU | others | barrier | bicycle | bus | car | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer |  truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation | 
----|:----------:| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :----------------------: | :---: | :------: | :------: |
bevformer_intern-s_occ|[Google Drive](https://drive.google.com/file/d/1LV9K8hrskKf51xY1wbqTKzK7WZmVXEV_/view?usp=sharing)| 25.11 | 6.93 | 35.57 | 10.40 | 35.97 | 41.23 | 13.72 | 20.30 | 21.10 | 18.34 | 19.18 | 28.64 | 49.82 | 30.74 | 31.00 | 27.44 | 19.29 | 17.29 | 
bevformer_base_occ|[Google Drive](https://drive.google.com/file/d/1NyoiosafAmne1qiABeNOPXR-P-y0i7_I/view?usp=share_link)| 23.67 | 5.03 | 38.79 | 9.98 | 34.41 | 41.09 | 13.24 | 16.50 | 18.15 | 17.83 | 18.66 | 27.70 | 48.95 | 27.73 | 29.08 | 25.38 | 15.41 | 14.46 | 


# CVPR2023 Occupancy Prediction Challenge Information

## Table of Contents
- [CVPR 2023 Occupancy Prediction Challenge](#cvpr-2023-occupancy-prediction-challenge)
  - [Introduction](#introduction)
  - [Task Definition](#task-definition)
    - [Rules for Occupancy Challenge](#rules-for-occupancy-challenge)
  - [Evaluation Metrics](#evaluation-metrics)
    - [mIoU](#miou)
    - [F Score](#f-score)
  - [Data](#data)
    - [Basic Information](#basic-information)
    - [Download](#download)
    - [Hierarchy](#hierarchy)
    - [Known Issues](#known-issues)
  - [Getting Started](#getting-started)
  - [Timeline](#challenge-timeline)
  - [Leaderboard](#leaderboard)
  - [License](#license)


## Introduction
Understanding the 3D surroundings including the background stuffs and foreground objects is important for autonomous driving. In the traditional 3D object detection task, a foreground object is represented by the 3D bounding box. However, the geometrical shape of the object is complex, which can not be represented by a simple 3D box, and the perception of the background is absent. The goal of this task is to predict the 3D occupancy of the scene. In this task, we provide a large-scale occupancy benchmark based on the nuScenes dataset. The benchmark is a voxelized representation of the 3D space, and the occupancy state and semantics of the voxel in 3D space are jointly estimated in this task. The complexity of this task lies in the dense prediction of 3D space given the surround-view image.

<p align="right">(<a href="#top">back to top</a>)</p>

## Task Definition
Given images from multiple cameras, the goal is to predict the current occupancy state and semantics of each voxel grid in the scene. The voxel state is predicted to be either free or occupied. If a voxel is occupied, its semantic class needs to be predicted, as well. Besides, we also provide a binary observed/unobserved mask for each frame. An observed voxel is defined as an invisible grid in the current camera observation, which is ignored in the evaluation stage.

### Rules for Occupancy Challenge
* We allow using annotations provided in the nuScenes dataset, and during inference, the input modality of the model should be camera only. 
* Other public/private datasets are not allowed in the challenge in any form (except ImageNet or MS-COCO pre-trained image backbone). 
* No future frame is allowed during inference.
* In order to check the compliance, we will ask the participants to provide technical reports to the challenge committee and the participant will be asked to provide a public talk about the method after winning the award.

<p align="right">(<a href="#top">back to top</a>)</p>

## Evaluation Metrics
Leaderboard ranking for this challenge is by the intersection-over-union (mIoU) over all classes. 
### mIoU

Let $C$ be he number of classes. 

$$
    mIoU=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$ , $FP_c$ , and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.

### F-Score
We also measure the F-score as the harmonic mean of the completeness $P_c$ and the accuracy $P_a$.

$$
    F-score=\left( \frac{P_a^{-1}+P_c^{-1}}{2} \right) ^{-1} ,
$$

where $P_a$ is the percentage of predicted voxels that are within a distance threshold to the ground truth voxels, and $P_c$ is the percentage of ground truth voxels that are within a distance threshold to the predicted voxels.

<p align="right">(<a href="#top">back to top</a>)</p>


## Data
<div id="top"  align="center">
<img src="./figs/mask.jpg">
</div>
<div id="top" align="center">
Figure 1. Semantic labels (left), visibility masks in the LiDAR (middle) and the camera (right) view. Grey voxels are unobserved in LiDAR view and white voxels are observed in the accumulative LiDAR view but unobserved in the current camera view.
</div>

### Basic Information
<div align="center">
  
| Type |  Info |
| :----: | :----: |
| mini            | 404 |
| train           | 28,130 |
| val             | 6,019 |
| test            | 6,006 |
| cameras         | 6 |
| voxel size      | 0.4m |
| range           | [-40m, -40m, -1m, 40m, 40m, 5.4m]|
| volume size     | [200, 200, 16]|
| #classes        | 0 - 17 |
  
</div>

- The dataset contains 18 classes. The definition of classes from 0 to 16 is the same as the [nuScenes-lidarseg](https://github.com/nutonomy/nuscenes-devkit/blob/fcc41628d41060b3c1a86928751e5a571d2fc2fa/python-sdk/nuscenes/eval/lidarseg/README.md) dataset. The label 17 category represents voxels that are not occupied by anything, which is named as `free`. Voxel semantics for each sample frame is given as `[semantics]` in the labels.npz. 

- <strong>How are the labels annotated?</strong> The ground truth labels of occupancy derive from accumulative LiDAR scans with human annotations. 
  - If a voxel reflects a LiDAR point, then it is assigned as the same semantic label as the LiDAR point;
  - If a LiDAR beam passes through a voxel in the air, the voxel is set to be `free`;
  - Otherwise, we set the voxel to be unknown, or unobserved. This happens due to the sparsity of the LiDAR or the voxel is occluded, e.g. by a wall. In the dataset, `[mask_lidar]` is a 0-1 binary mask, where 0's represent unobserved voxels. As shown in Fig.1(b), grey voxels are unobserved. Due to the limitation of the visualization tool, we only show unobserved voxels at the same height as the ground. 

- <strong>Camera visibility.</strong> Note that the installation positions of LiDAR and cameras are different, therefore, some observed voxels in the  LiDAR view are not seen by the cameras. Since we focus on a vision-centric task, we provide a binary voxel mask `[mask_camera]`, indicating whether the voxels are observed or not in the current camera view. As shown in Fig.1(c), white voxels are observed in the accumulative LiDAR view but unobserved in the current camera view.

- Both `[mask_lidar]` and `[mask_camera]` masks are optional for training. Participants do not need to predict the masks. Only `[mask_camera]` is used for evaluation; the unobserved voxels are not involved during calculating the F-score and mIoU.


### Download
The files mentioned below can also be downloaded via <img src="https://user-images.githubusercontent.com/29263416/222076048-21501bac-71df-40fa-8671-2b5f8013d2cd.png" alt="OpenDataLab" width="18"/>[OpenDataLab](https://opendatalab.com/CVPR2023-3D-Occupancy/download).It is recommended to use provided [command line interface](https://opendatalab.com/CVPR2023-3D-Occupancy/cli) for acceleration.

| Subset | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Size |
| :---: | :---: | :---: | :---: |
| mini | [data](https://drive.google.com/drive/folders/1ksWt4WLEqOxptpWH2ZN-t1pjugBhg3ME?usp=share_link) | [data](https://pan.baidu.com/s/1IvOoJONwzKBi32Ikjf8bSA?pwd=5uv6)  | approx. 440M |
| trainval  | [data](https://drive.google.com/drive/folders/1JObO75iTA2Ge5fa8D3BWC8R7yIG8VhrP?usp=share_link) | [data](https://pan.baidu.com/s/1_4yE0__UDIJS8JtBSB0Bpg?pwd=li5h) | approx. 32G |
| test | coming soon | coming soon | ~ |

* Mini and trainval data contain three parts -- `imgs`, `gts` and `annotations`. The `imgs` datas have the same hierarchy with the image samples in the original nuScenes dataset.


### Hierarchy
The hierarchy of folder `Occpancy3D-nuScenes-V1.0/` is described below:
```
└── Occpancy3D-nuScenes-V1.0
    |
    ├── mini
    |
    ├── trainval
    |   ├── imgs
    |   |   ├── CAM_BACK
    |   |   |   ├── n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525.jpg
    |   |   |   └── ...
    |   |   ├── CAM_BACK_LEFT
    |   |   |   ├── n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg
    |   |   |   └── ...
    |   |   └── ...
    |   |     
    |   ├── gts  
    |   |   ├── [scene_name]
    |   |   |   ├── [frame_token]
    |   |   |   |   └── labels.npz
    |   |   |   └── ...
    |   |   └── ...
    |   |
    |   └── annotations.json
    |
    └── test
        ├── imgs
        └── annotations.json

```
- `imgs/` contains images captured by various cameras.
- `gts/` contains the ground truth of each sample. `[scene_name]` specifies a sequence of frames, and `[frame_token]` specifies a single frame in a sequence.
- `annotations.json` contains meta infos of the dataset.
- `labels.npz` contains `[semantics]`, `[mask_lidar]`, and `[mask_camera]` for each frame. 

```
annotations {
    "train_split": ["scene-0001", ...],                         <list> -- training dataset split by scene_name
    "val_split": list ["scene-0003", ...],                      <list> -- validation dataset split by scene_name
    "scene_infos" {                                             <dict> -- meta infos of the scenes    
        [scene_name]: {                                         <str> -- name of the scene.  
            [frame_token]: {                                    <str> -- samples in a scene, ordered by time
                    "timestamp":                                <str> -- timestamp (or token), unique by sample
                    "camera_sensor": {                          <dict> -- meta infos of the camera sensor
                        [cam_token]: {                          <str> -- token of the camera
                            "img_path":                         <str> -- corresponding image file path, *.jpg
                            "intrinsic":                        <float> [3, 3] -- intrinsic camera calibration
                            "extrinsic":{                       <dict> -- extrinsic parameters of the camera
                                "translation":                  <float> [3] -- coordinate system origin in meters
                                "rotation":                     <float> [4] -- coordinate system orientation as quaternion
                            }   
                            "ego_pose": {                       <dict> -- vehicle pose of the camera
                                "translation":                  <float> [3] -- coordinate system origin in meters
                                "rotation":                     <float> [4] -- coordinate system orientation as quaternion
                            }                
                        },
                        ...
                    },
                    "ego_pose": {                               <dict> -- vehicle pose
                        "translation":                          <float> [3] -- coordinate system origin in meters
                        "rotation":                             <float> [4] -- coordinate system orientation as quaternion
                    },
                    "gt_path":                                  <str> -- corresponding 3D voxel gt path, *.npz
                    "next":                                     <str> -- frame_token of the previous keyframe in the scene 
                    "prev":                                     <str> -- frame_token of the next keyframe in the scene
                }
            ]             
        }
    }
}
```

### Known Issues
- Nuscene ([issues-721](https://github.com/nutonomy/nuscenes-devkit/issues/721)) lacks translation in the z-axis, which makes it hard to recover accurate 6d localization and would lead to the misalignment of point clouds while accumulating them over whole scenes. Ground stratification occurs in several data.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

We provide a baseline model based on [BEVFormer](https://github.com/fundamentalvision/BEVFormer).

Please refer to [getting_started](docs/getting_started.md) for details.

<p align="right">(<a href="#top">back to top</a>)</p>


## Challenge Timeline
- Pending - Challenge Period Open.
- Jun 01, 2023 - Challenge Period End.
- Jun 03, 2023 - Finalist Notification.
- Jun 10, 2023 - Technical Report Deadline.
- Jun 12, 2023 - Winner Announcement.
<p align="right">(<a href="#top">back to top</a>)</p>


## Leaderboard 
To be released.

<p align="right">(<a href="#top">back to top</a>)</p>

## License
Before using the dataset, you should register on the website and agree to the terms of use of the [nuScenes](https://www.nuscenes.org/nuscenes).
All code within this repository is under [Apache License 2.0](./LICENSE).

<p align="right">(<a href="#top">back to top</a>)</p>
