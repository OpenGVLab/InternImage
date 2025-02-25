# InternImage for Object Detection

This folder contains the implementation of the InternImage for object detection.

Our detection code is developed on top of [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1).

<!-- TOC -->

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Released Models](#released-models)
- [Evaluation](#evaluation)
- [Training](#training)
- [Manage Jobs with Slurm](#manage-jobs-with-slurm)
- [Export](#export)

<!-- TOC -->

## Installation

- Clone this repository:

```bash
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.9
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install `torch==1.11` with `CUDA==11.3`:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip.

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
```

- Install `timm`, `mmcv-full` and \`mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
# Please use a version of numpy lower than 2.0
pip install numpy==1.26.4
pip install pydantic==1.10.13
pip install yapf==0.40.1
```

- Compile CUDA operators

Before compiling, please use the `nvcc -V` command to check whether your `nvcc` version matches the CUDA version of PyTorch.

```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

- You can also install the operator using precompiled `.whl` files
  [DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

## Data Preparation

Prepare datasets according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

## Released Models

<details open>
<summary> Dataset: COCO </summary>
<br>
<div>

|   method   |    backbone    | schd | box mAP | mask mAP | #param | FLOPs |                                     Config                                     |                                                                                                                         Download                                                                                                                         |
| :--------: | :------------: | :--: | :-----: | :------: | :----: | :---: | :----------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mask R-CNN | InternImage-T  |  1x  |  47.2   |   42.5   |  49M   | 270G  |        [config](./configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py)         |                [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.log.json)                |
| Mask R-CNN | InternImage-T  |  3x  |  49.1   |   43.7   |  49M   | 270G  |        [config](./configs/coco/mask_rcnn_internimage_t_fpn_3x_coco.py)         |                [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.log.json)                |
| Mask R-CNN | InternImage-S  |  1x  |  47.8   |   43.3   |  69M   | 340G  |        [config](./configs/coco/mask_rcnn_internimage_s_fpn_1x_coco.py)         |                [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.log.json)                |
| Mask R-CNN | InternImage-S  |  3x  |  49.7   |   44.5   |  69M   | 340G  |        [config](./configs/coco/mask_rcnn_internimage_s_fpn_3x_coco.py)         |                [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.log.json)                |
| Mask R-CNN | InternImage-B  |  1x  |  48.8   |   44.0   |  115M  | 501G  |        [config](./configs/coco/mask_rcnn_internimage_b_fpn_1x_coco.py)         |                [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_1x_coco.log.json)                |
| Mask R-CNN | InternImage-B  |  3x  |  50.3   |   44.8   |  115M  | 501G  |        [config](./configs/coco/mask_rcnn_internimage_b_fpn_3x_coco.py)         |                [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_3x_coco.log.json)                |
|  Cascade   | InternImage-L  |  1x  |  54.9   |   47.7   |  277M  | 1399G |         [config](./configs/coco/cascade_internimage_l_fpn_1x_coco.py)          |                                                                         [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_1x_coco.pth)                                                                          |
|  Cascade   | InternImage-L  |  3x  |  56.1   |   48.5   |  277M  | 1399G |         [config](./configs/coco/cascade_internimage_l_fpn_3x_coco.py)          |                  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_l_fpn_3x_coco.log.json)                  |
|  Cascade   | InternImage-XL |  1x  |  55.3   |   48.1   |  387M  | 1782G |         [config](./configs/coco/cascade_internimage_xl_fpn_1x_coco.py)         |                 [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_1x_coco.log.json)                 |
|  Cascade   | InternImage-XL |  3x  |  56.2   |   48.8   |  387M  | 1782G |         [config](./configs/coco/cascade_internimage_xl_fpn_3x_coco.py)         |                 [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.log.json)                 |
|    DINO    | InternImage-T  |  1x  |  53.9   |    -     |  49M   |   -   |  [config](./configs/coco/dino_4scale_internimage_t_1x_coco_layer_wise_lr.py)   |                    [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_t_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_t_1x_coco.json)                    |
|    DINO    | InternImage-L  |  1x  |  57.6   |    -     |  241M  |   -   | [config](./configs/coco/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.log.json) |
|    DINO    | InternImage-H  |  -   |  65.0   |    -     | 2.18B  |   -   |   [config](./configs/coco/dino_4scale_cbinternimage_h_objects365_coco_ss.py)   |                                                                    [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_coco.pth)                                                                     |

</div>

</details>

<details open>
<summary> Dataset: LVIS </summary>
<br>
<div>

| method |   backbone    | minival (ss) | val (ss/ms) | #param |                                       Config                                       |                                                     Download                                                      |
| :----: | :-----------: | :----------: | :---------: | :----: | :--------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
|  DINO  | InternImage-H |     65.8     | 62.3 / 63.2 | 2.18B  | [config](./configs/lvis/dino_4scale_cbinternimage_h_objects365_lvis_minival_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_lvis.pth) |

</div>

</details>

<details open>
<summary> Dataset: OpenImages </summary>
<br>
<div>

| method |   backbone    | mAP (ss) | #param |                                         Config                                         |                                                        Download                                                         |
| :----: | :-----------: | :------: | :----: | :------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
|  DINO  | InternImage-H |   74.1   | 2.18B  | [config](./configs/openimages/dino_4scale_cbinternimage_h_objects365_openimages_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_openimages.pth) |

</div>

</details>

## Evaluation

To evaluate our `InternImage` on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm
```

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py pretrained/mask_rcnn_internimage_t_fpn_1x_coco.pth --eval bbox segm
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/coco/mask_rcnn_internimage_b_fpn_1x_coco.py pretrained/mask_rcnn_internimage_b_fpn_1x_coco.py 8 --eval bbox segm
```

## Training

To train an `InternImage` on COCO, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node, run:

```bash
sh dist_train.sh configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py 8
```

## Manage Jobs with Slurm

For example, to train `InternImage-L` with 32 GPU on 4 node, run:

```bash
GPUS=32 sh slurm_train.sh <partition> <job-name> configs/coco/cascade_internimage_xl_fpn_3x_coco.py work_dirs/cascade_internimage_xl_fpn_3x_coco
```

## Export

Install `mmdeploy` at first:

```shell
pip install mmdeploy==0.14.0
```

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
