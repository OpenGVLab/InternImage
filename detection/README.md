# InternImage for Object Detection and Instance Segmentation

This folder contains the implementation of the InternImage for object detection and instance segmentation. 

**Note:** Our detection code is developed on top of [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1).


## Usage

### Install
See "install" section in segmentation/README.md. 

### Inference
To run inference on a single image or multiple images, you can run as below.

**Note:** If you specify an image containing directory instead of a single image, it will process all the images in the directory.:
```
python image_demo.py path/to/your/image/or/directory <config-file> <checkpoint>
e.g. 
python image_demo.py \
  data/coco/val2014/COCO_val2014_000000000042.jpg \
  configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py \
  ../checkpoint_dir/det/mask_rcnn_internimage_t_fpn_1x_coco.pth 
```

### Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).


### Evaluation

To evaluate our `InternImage` on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm
```

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py checkpoint_dir/det/mask_rcnn_internimage_t_fpn_1x_coco.pth --eval bbox segm
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/coco/mask_rcnn_internimage_b_fpn_1x_coco.py checkpoint_dir/det/mask_rcnn_internimage_b_fpn_1x_coco.py 8 --eval bbox segm
```

### Training on COCO

To train an `InternImage` on COCO, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node, run:

```bash
sh dist_train.sh configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py 8
```

### Manage Jobs with Slurm

For example, to train `InternImage-L` with 32 GPU on 4 node, run:

```bash
GPUS=32 sh slurm_train.sh <partition> <job-name> configs/coco/cascade_internimage_xl_fpn_3x_coco.py work_dirs/cascade_internimage_xl_fpn_3x_coco
```

### Export

To export a detection model from PyTorch to TensorRT, run:
```shell
MODEL="model_name"
CKPT_PATH="/path/to/model/ckpt.pth"

python deploy.py \
    "./deploy/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py" \
    "./configs/coco/${MODEL}.py" \
    "${CKPT_PATH}" \ 
    "./deploy/demo.jpg" \
    --work-dir "./work_dirs/mmdet/instance-seg/${MODEL}" \
    --device cuda \
    --dump-info
```

For example, to export `mask_rcnn_internimage_t_fpn_1x_coco` from PyTorch to TensorRT, run:
```shell
MODEL="mask_rcnn_internimage_t_fpn_1x_coco"
CKPT_PATH="/path/to/model/ckpt/mask_rcnn_internimage_t_fpn_1x_coco.pth"

python deploy.py \
    "./deploy/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py" \
    "./configs/coco/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.jpg" \
    --work-dir "./work_dirs/mmdet/instance-seg/${MODEL}" \
    --device cuda \
    --dump-info
```
