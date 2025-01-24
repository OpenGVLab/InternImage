# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation.

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

<!-- TOC -->

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Released Models](#released-models)
- [Evaluation](#evaluation)
- [Training](#training)
- [Manage Jobs with Slurm](#manage-jobs-with-slurm)
- [Image Demo](#image-demo)
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

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Released Models

<details open>
<summary> Dataset: ADE20K </summary>
<br>
<div>

|   method    |    backbone    | resolution | mIoU (ss/ms) | #param | FLOPs |                                       Config                                        |                                                                                                                       Download                                                                                                                       |
| :---------: | :------------: | :--------: | :----------: | :----: | :---: | :---------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   UperNet   | InternImage-T  |  512x512   | 47.9 / 48.1  |  59M   | 944G  |         [config](./configs/ade20k/upernet_internimage_t_512_160k_ade20k.py)         |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_t_512_160k_ade20k.log.json)              |
|   UperNet   | InternImage-S  |  512x512   | 50.1 / 50.9  |  80M   | 1017G |         [config](./configs/ade20k/upernet_internimage_s_512_160k_ade20k.py)         |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_s_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_s_512_160k_ade20k.log.json)              |
|   UperNet   | InternImage-B  |  512x512   | 50.8 / 51.3  |  128M  | 1185G |         [config](./configs/ade20k/upernet_internimage_b_512_160k_ade20k.py)         |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_b_512_160k_ade20k.log.json)              |
|   UperNet   | InternImage-L  |  640x640   | 53.9 / 54.1  |  256M  | 2526G |         [config](./configs/ade20k/upernet_internimage_l_640_160k_ade20k.py)         |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_640_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_l_640_160k_ade20k.log.json)              |
|   UperNet   | InternImage-XL |  640x640   | 55.0 / 55.3  |  368M  | 3142G |        [config](./configs/ade20k/upernet_internimage_xl_640_160k_ade20k.py)         |             [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_640_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_xl_640_160k_ade20k.log.json)             |
|   UperNet   | InternImage-H  |  896x896   | 59.9 / 60.3  | 1.12B  | 3566G |         [config](./configs/ade20k/upernet_internimage_h_896_160k_ade20k.py)         |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_h_896_160k_ade20k.log.json)              |
| Mask2Former | InternImage-H  |  896x896   | 62.6 / 62.9  | 1.31B  | 4635G | [config](./configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.log.json) |

</div>

</details>

<details>
<summary> Dataset: Cityscapes </summary>
<br>
<div>

|    method     |    backbone    | resolution | mIoU (ss/ms)  | #params | FLOPs |                                             Config                                             |                                                                                                                                 Download                                                                                                                                 |
| :-----------: | :------------: | :--------: | :-----------: | :-----: | :---: | :--------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    UperNet    | InternImage-T  |  512x1024  | 82.58 / 83.40 |   59M   | 1889G |        [config](./configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py)        |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_t_512x1024_160k_cityscapes.log.json)               |
|    UperNet    | InternImage-S  |  512x1024  | 82.74 / 83.45 |   80M   | 2035G |        [config](./configs/cityscapes/upernet_internimage_s_512x1024_160k_cityscapes.py)        |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_s_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_s_512x1024_160k_cityscapes.log.json)               |
|    UperNet    | InternImage-B  |  512x1024  | 83.18 / 83.97 |  128M   | 2369G |        [config](./configs/cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.py)        |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_b_512x1024_160k_cityscapes.log.json)               |
|    UperNet    | InternImage-L  |  512x1024  | 83.68 / 84.41 |  256M   | 3234G |        [config](./configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py)        |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_l_512x1024_160k_cityscapes.log.json)               |
|   UperNet\*   | InternImage-L  |  512x1024  | 85.94 / 86.22 |  256M   | 3234G |   [config](./configs/cityscapes/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.py)   |    [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.pth)  \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.log.json)     |
|    UperNet    | InternImage-XL |  512x1024  | 83.62 / 84.28 |  368M   | 4022G |       [config](./configs/cityscapes/upernet_internimage_xl_512x1024_160k_cityscapes.py)        |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_xl_512x1024_160k_cityscapes.log.json)              |
|   UperNet\*   | InternImage-XL |  512x1024  | 86.20 / 86.42 |  368M   | 4022G |  [config](./configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py)   |    [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.log.json)    |
|  SegFormer\*  | InternImage-L  |  512x1024  | 85.16 / 85.67 |  220M   | 1580G |  [config](./configs/cityscapes/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py)  |   [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.log.json)   |
|  SegFormer\*  | InternImage-XL |  512x1024  | 85.41 / 85.93 |  330M   | 2364G | [config](./configs/cityscapes/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.py)  |  [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.log.json)  |
| Mask2Former\* | InternImage-H  | 1024x1024  | 86.37 / 86.96 |  1094M  | 7878G | [config](./configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.log.json) |

\* denotes the model is trained using extra Mapillary dataset.

</div>

</details>

<details>
<summary> Dataset: COCO-Stuff-164K </summary>
<br>
<div>

|   method    |   backbone    | resolution | mIoU (ss) | #params | FLOPs |                                        Config                                         |                                                                                                                    Download                                                                                                                    |
| :---------: | :-----------: | :--------: | :-------: | :-----: | :---: | :-----------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mask2Former | InternImage-H |  896x896   |   52.6    |  1.31B  | 4635G | [config](./configs/coco_stuff164k/mask2former_internimage_h_896_80k_cocostuff164k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff164k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_896_80k_cocostuff164k.log.json) |

</div>

</details>

<details>
<summary> Dataset: COCO-Stuff-10K </summary>
<br>
<div>

|   method    |   backbone    | resolution |  mIoU (ss)  | #params | FLOPs |                                            Config                                            |                                                                                                                           Download                                                                                                                           |
| :---------: | :-----------: | :--------: | :---------: | :-----: | :---: | :------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mask2Former | InternImage-H |  512x512   | 59.2 / 59.6 |  1.28B  | 1528G | [config](./configs/coco_stuff10k/mask2former_internimage_h_512_40k_cocostuff164k_to_10k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_512_40k_cocostuff164k_to_10k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_512_40k_cocostuff164k_to_10k.log.json) |

</div>

</details>

## Evaluation

To evaluate our `InternImage` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_internimage_t_512_160k_ade20k.py pretrained/upernet_internimage_t_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/ade20k/upernet_internimage_b_512_160k_ade20k.py pretrained/upernet_internimage_b_512_160k_ade20k.pth 8 --eval mIoU
```

## Training

To train an `InternImage` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_internimage_t_512_160k_ade20k.py 8
```

## Manage Jobs with Slurm

For example, to train `InternImage-XL` with 8 GPU on 1 node (total batch size 16), run:

```bash
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/ade20k/upernet_internimage_xl_640_160k_ade20k.py
```

## Image Demo

To inference a single/multiple image like this.
If you specify image containing directory instead of a single image, it will process all the images in the directory.

```
CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  data/ade/ADEChallengeData2016/images/validation/ADE_val_00000591.jpg \
  configs/ade20k/upernet_internimage_t_512_160k_ade20k.py  \
  checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth  \
  --palette ade20k
```

## Export

Install `mmdeploy` at first:

```shell
pip install mmdeploy==0.14.0
```

To export a segmentation model from PyTorch to TensorRT, run:

```shell
MODEL="model_name"
CKPT_PATH="/path/to/model/ckpt.pth"

python deploy.py \
    "./deploy/configs/mmseg/segmentation_tensorrt_static-512x512.py" \
    "./configs/ade20k/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.png" \
    --work-dir "./work_dirs/mmseg/${MODEL}" \
    --device cuda \
    --dump-info
```

For example, to export `upernet_internimage_t_512_160k_ade20k` from PyTorch to TensorRT, run:

```shell
MODEL="upernet_internimage_t_512_160k_ade20k"
CKPT_PATH="/path/to/model/ckpt/upernet_internimage_t_512_160k_ade20k.pth"

python deploy.py \
    "./deploy/configs/mmseg/segmentation_tensorrt_static-512x512.py" \
    "./configs/ade20k/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.png" \
    --work-dir "./work_dirs/mmseg/${MODEL}" \
    --device cuda \
    --dump-info
```
