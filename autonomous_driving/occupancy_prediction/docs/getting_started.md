## Installation
Follow https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md to prepare the environment.

## Preparing Dataset
1. Download the gts and annotations.json we provided. You can download our imgs.tar.gz or using the original sample files of the nuScenes dataset.

2. Download the CAN bus expansion data and maps [HERE](https://www.nuscenes.org/download).

3. Organize your folder structure as below：
```
Occupancy3D
├── projects/
├── tools/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── occ3d-nus/
│   │   ├── maps/
│   │   ├── samples/     # You can download our imgs.tar.gz or using the original sample files of the nuScenes dataset
│   │   ├── v1.0-trainval/
│   │   ├── gts/
│   │   │── annotations.json
```


4. Generate the info files for training and validation:
```
python tools/create_data.py occ --root-path ./data/occ3d-nus --out-dir ./data/occ3d-nus --extra-tag occ --version v1.0-trainval --canbus ./data --occ-path ./data/occ3d-nus
``` 

## Training
```
./tools/dist_train.sh projects/configs/bevformer/bevformer_base_occ.py 8
```

## Testing
```
./tools/dist_test.sh projects/configs/bevformer/bevformer_base_occ.py work_dirs/bevformer_base_occ/epoch_24.pth 8
```
You can evaluate the F-score at the same time by adding `--eval_fscore`.


### Performance

model name|weight| mIoU | others | barrier | bicycle | bus | car | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer |  truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation | 
----|:----------:| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :----------------------: | :---: | :------: | :------: |
bevformer_base_occ|[Google Drive](https://drive.google.com/file/d/1NyoiosafAmne1qiABeNOPXR-P-y0i7_I/view?usp=share_link)| 23.67 | 5.03 | 38.79 | 9.98 | 34.41 | 41.09 | 13.24 | 16.50 | 18.15 | 17.83 | 18.66 | 27.7 | 48.95 | 27.73 | 29.08 | 25.38 | 15.41 | 14.46 | 


