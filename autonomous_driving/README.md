<div id="top" align="center">

# InternImage for CVPR 2023 Workshop on End-to-End Autonomous Driving
 </div>



## 1. InternImage-based Baseline for CVPR23 Occupancy Prediction Challenge
We achieve an improvement of 1.44 in MIOU baseline by leveraging the InterImage-based model.

model name|weight| mIoU | others | barrier | bicycle | bus | car | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer |  truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation | 
----|:----------:| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :----------------------: | :---: | :------: | :------: |
bevformer_intern-s_occ|[Google Drive](https://drive.google.com/file/d/1LV9K8hrskKf51xY1wbqTKzK7WZmVXEV_/view?usp=sharing)| 25.11 | 6.93 | 35.57 | 10.40 | 35.97 | 41.23 | 13.72 | 20.30 | 21.10 | 18.34 | 19.18 | 28.64 | 49.82 | 30.74 | 31.00 | 27.44 | 19.29 | 17.29 | 
bevformer_base_occ|[Google Drive](https://drive.google.com/file/d/1NyoiosafAmne1qiABeNOPXR-P-y0i7_I/view?usp=share_link)| 23.67 | 5.03 | 38.79 | 9.98 | 34.41 | 41.09 | 13.24 | 16.50 | 18.15 | 17.83 | 18.66 | 27.70 | 48.95 | 27.73 | 29.08 | 25.38 | 15.41 | 14.46 | 

### Get Started
please refer to [README.md](./occupancy_prediction/README.md)


## 2. InternImage-based Baseline for Online HD Map Construction Challenge For Autonomous Driving
By incorporating the InterImage-based model, we observe an enhancement of 6.56 in mAP baseline.

model name|weight|$\mathrm{mAP}$ | $\mathrm{AP}_{pc}$ | $\mathrm{AP}_{div}$ | $\mathrm{AP}_{bound}$ | 
----|:----------:| :--: | :--: | :--: | :--: | 
vectormapnet_intern|[Checkpoint](https://github.com/OpenGVLab/InternImage/releases/download/track_model/vectormapnet_internimage.pth) | 49.35 | 45.05 | 56.78 | 46.22 | 
vectormapnet_base|[Google Drive](https://drive.google.com/file/d/16D1CMinwA8PG1sd9PV9_WtHzcBohvO-D/view) | 42.79 | 37.22 | 50.47	 | 40.68 | 

### Get Started
please refer to [README.md](Online-HD-Map-Construction/README.md)


## 3. InternImage-based Baseline for CVPR23 OpenLane-V2 Challenge
Through the implementation of the InterImage-based model, we achieve an advancement of 0.009 in F-score baseline.


|             | OpenLane-V2 Score | DET<sub>l</sub> | DET<sub>t</sub> | TOP<sub>ll</sub> | TOP<sub>lt</sub> | F-Score |
|-------------|-------------------|-----------------|-----------------|------------------|------------------|---------|
| base r50    | 0.292             | 0.183           | 0.457           | 0.022            | 0.143            | 0.215   |
| InternImage | 0.325             | 0.194           | 0.537           | 0.02             | 0.17             | 0.224   |

### Get Started
please refer to [README.md](./openlane-v2/README.md)




