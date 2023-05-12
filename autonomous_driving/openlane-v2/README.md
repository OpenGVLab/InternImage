## InternImage-based Baseline for CVPR23 OpenLane-V2 Challenge!!!!

We improve our baseline with a more powerful image backbone: **InaternImage**, which shows its excellent ability within a series of leaderboards and benchmarks, such as *COCO* and *nuScenes*.


#### 1. Requirements
```bash
python>=3.8
torch==1.11
mmcv-full>=1.5.2
mmdet==2.28.0
mmsegmentation==0.29.1
timm
```


### 2. Install DCNv3 for InternImage
```bash
cd plugin/mmdet3d/baseline/models/backbones/ops_dcnv3
bash make.sh # requires torch>=1.10
```

### 3. Train with InternImage-Small

```bash
./tools/dist_train.sh plugin/mmdet3d/configs/internimage-s.py 8
```

Notes: InternImage provides abundant pre-trained model weights that can be used!!!


### 4. Performance compared to baseline

|             | OpenLane-V2 Score | DET<sub>l</sub> | DET<sub>t</sub> | TOP<sub>ll</sub> | TOP<sub>lt</sub> | F-Score |
|-------------|-------------------|-----------------|-----------------|------------------|------------------|---------|
| base r50    | 0.292             | 0.183           | 0.457           | 0.022            | 0.143            | 0.215   |
| InternImage | 0.325             | 0.194           | 0.537           | 0.02             | 0.17             | 0.224   |


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
