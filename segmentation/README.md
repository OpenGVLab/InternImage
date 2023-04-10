# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.7 -y
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.5.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
- You can also install the operator using .whl files
[DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

### Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Evaluation

To evaluate our `InternImage` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_internimage_t_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/ade20k/upernet_internimage_b_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_b_512_160k_ade20k.pth 8 --eval mIoU
```

### Training

To train an `InternImage` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_internimage_t_512_160k_ade20k.py 8
```

### Manage Jobs with Slurm

For example, to train `InternImage-XL` with 8 GPU on 1 node (total batch size 16), run:

```bash
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/ade20k/upernet_internimage_xl_640_160k_ade20k.py
```

### Image Demo
To inference a single image like this:
```
CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  data/ade/ADEChallengeData2016/images/validation/ADE_val_00000591.jpg \
  configs/ade20k/upernet_internimage_t_512_160k_ade20k.py  \
  checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth  \
  --palette ade20k 
```

### Export

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
