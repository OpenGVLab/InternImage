
## InternImage-based Baseline for CVPR23 Occupancy Prediction Challenge!!!!

We improve our baseline with a more powerful image backbone: **InternImage**, which shows its excellent ability within a series of leaderboards and benchmarks, such as *COCO* and *nuScenes*.


#### 1. Requirements
```bash
python>=3.8
torch==1.12 # recommend
mmcv-full>=1.5.0
mmdet==2.24.0
mmsegmentation==0.24.0
timm
numpy==1.22
mmdet3d==0.18.1 # recommend
```


### 2. Install DCNv3 for InternImage
```bash
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
bash make.sh # requires torch>=1.10
```

### 3. Train with InternImage-Small

```bash
./tools/dist_train.sh projects/configs/bevformer/bevformer_intern-s_occ.py 8 # consumes less than 14G memory
```

Notes: InatenImage provides abundant pre-trained model weights that can be used!!!


### 4. Performance compared to baseline

model name|weight| mIoU | others | barrier | bicycle | bus | car | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer |  truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation | 
----|:----------:| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :----------------------: | :---: | :------: | :------: |
bevformer_intern-s_occ|[Google Drive](https://drive.google.com/file/d/1LV9K8hrskKf51xY1wbqTKzK7WZmVXEV_/view?usp=sharing)| 25.11 | 6.93 | 35.57 | 10.40 | 35.97 | 41.23 | 13.72 | 20.30 | 21.10 | 18.34 | 19.18 | 28.64 | 49.82 | 30.74 | 31.00 | 27.44 | 19.29 | 17.29 | 
bevformer_base_occ|[Google Drive](https://drive.google.com/file/d/1NyoiosafAmne1qiABeNOPXR-P-y0i7_I/view?usp=share_link)| 23.67 | 5.03 | 38.79 | 9.98 | 34.41 | 41.09 | 13.24 | 16.50 | 18.15 | 17.83 | 18.66 | 27.70 | 48.95 | 27.73 | 29.08 | 25.38 | 15.41 | 14.46 | 



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
