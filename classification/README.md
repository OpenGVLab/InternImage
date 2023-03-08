# InternImage for Image Classification

This folder contains the implementation of the InternImage for image classification.

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
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
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

- Compiling CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train.txt`, `val.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 meta_data/val.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 meta_data/train.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
- For ImageNet-22K dataset, make a folder named `fall11_whole` and move all images to labeled sub-folders in this
  folder. Then download the train-val split
  file ([ILSVRC2011fall_whole_map_train.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_train.txt)
  & [ILSVRC2011fall_whole_map_val.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt))
  , and put them in the parent directory of `fall11_whole`. The file structure should look like:

  ```bash
    $ tree imagenet22k/
    imagenet22k/
    └── fall11_whole
        ├── n00004475
        ├── n00005787
        ├── n00006024
        ├── n00006484
        └── ...
  ```

### Evaluation

To evaluate a pre-trained `InternImage` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the `InternImage-B` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/internimage_b_1k_224.yaml --resume internimage_b_1k_224.pth --data-path <imagenet-path>
```

### Training from scratch on ImageNet-1K

To train an `InternImage` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

### Manage jobs with Srun.

For example, to train `InternImage` with 8 GPU on a single node for 300 epochs, run:

`InternImage-T`:

```bash
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --resume internimage_t_1k_224.pth --eval
```

`InternImage-S`:

```bash
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_s_1k_224.yaml --resume internimage_s_1k_224.pth --eval
```

`InternImage-XL`:

```bash
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_xl_22kto1k_384.pth --resume internimage_xl_22kto1k_384.pth --eval
```

<!-- 
### Test pretrained model on ImageNet-22K

For example, to evaluate the `InternImage-L-22k`:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg configs/internimage_xl_22k_192to384.yaml --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory>] \
--resume internimage_xl_22k_192to384.pth --eval
``` -->

<!-- ### Fine-tuning from a ImageNet-22K pre-trained model

For example, to fine-tune a `InternImage-XL-22k` model pre-trained on ImageNet-22K:

```bashs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_image_.yaml --pretrained intern_image_b.pth --eval
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/.yaml --pretrained swin_base_patch4_window7_224_22k.pth \
--data-path <imagenet-path> --batch-size 64 --accumulation-steps 2 [--use-checkpoint]
``` -->

### Export

To export `InternImage-T` from PyTorch to ONNX, run:
```shell
python export.py --model_name internimage_t_1k_224 --ckpt_dir /path/to/ckpt/dir --onnx
```

To export `InternImage-T` from PyTorch to TensorRT, run:
```shell
python export.py --model_name internimage_t_1k_224 --ckpt_dir /path/to/ckpt/dir --trt
```
