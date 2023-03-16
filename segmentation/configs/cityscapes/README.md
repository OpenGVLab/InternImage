# Cityscapes

Introduced by Cordts et al. in [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://paperswithcode.com/paper/the-cityscapes-dataset-for-semantic-urban).

Cityscapes is a large-scale database which focuses on semantic understanding of urban street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories (flat surfaces, humans, vehicles, constructions, objects, nature, sky, and void). The dataset consists of around 5000 fine annotated images and 20000 coarse annotated ones. Data was captured in 50 cities during several months, daytimes, and good weather conditions. It was originally recorded as video so the frames were manually selected to have the following features: large number of dynamic objects, varying scene layout, and varying background.

## Model Zoo

### UperNet + InternImage

| backbone       | resolution |  mIoU (ss/ms) | train speed | train time | #params | FLOPs | Config | Download                                                                      |
|:--------------:|:----------:|:------------:|:-----------:|:----------:|:-------:|:-----:|:----:|:----:|
| InternImage-T  | 512x1024   |   82.58 / 83.40    | 0.32s / iter       | 14.5h      | 59M     | 1889G | [config](./upernet_internimage_t_512x1024_160k_cityscapes.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_t_512x1024_160k_cityscapes.log.json) |
| InternImage-S  | 512x1024   |   82.74 / 83.45    | 0.36s / iter       | 16.5h      | 80M     | 2035G | [config](./upernet_internimage_s_512x1024_160k_cityscapes.py) |[ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_s_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_s_512x1024_160k_cityscapes.log.json)  |
| InternImage-B  | 512x1024   |   83.18 / 83.97    | 0.39s / iter       | 17h        | 128M    | 2369G | [config](./upernet_internimage_b_512x1024_160k_cityscapes.py) |[ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_b_512x1024_160k_cityscapes.log.json)  |
| InternImage-L  | 512x1024   |    83.68 / 84.41   | 0.50s / iter       | 23h        | 256M    | 3234G | [config](./upernet_internimage_l_512x1024_160k_cityscapes.py) |[ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_l_512x1024_160k_cityscapes.log.json)  |
| InternImage-XL | 512x1024   |    83.62 / 84.28   | 0.56s / iter       | 26h       | 368M    | 4022G | [config](./upernet_internimage_xl_512x1024_160k_cityscapes.py) |[ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_xl_512x1024_160k_cityscapes.log.json) |

- Training speed is measured with A100 GPU.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.

### UperNet + InternImage (with additional data)

Mapillary 80k + Cityscapes (w/ coarse data) 160k 

| backbone       | resolution |  mIoU (ss/ms) | train speed | train time | #params | FLOPs | Config | Download     |
|:--------------:|:----------:|:------------:|:-----------:|:-----------:|:-------:|:-----:|:------:|:------------:|
| InternImage-L  | 512x1024   | 85.94 / 86.22  | 0.50s / iter | 23h    | 256M  | 3234G | [config](./upernet_internimage_l_512x1024_160k_mapillary2cityscapes.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.pth)  \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.log.json)  |
| InternImage-XL | 512x1024   | 86.20 / 86.42  | 0.56s / iter | 26h    | 368M  | 4022G | [config](./upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.log.json) |

### SegFormerHead + InternImage (with additional data)

Mapillary 80k + Cityscapes (w/ coarse data) 160k

| backbone       | resolution |  mIoU (ss/ms) | train speed | train time | #params | FLOPs | Config | Download |
|:--------------:|:----------:|:------------:|:-----------:|:-----------:|:-------:|:-----:|:-----:|:---------:|
| InternImage-L  | 512x1024   | 85.16 / 85.67  | 0.37s / iter       | 17h        | 220M    | 1580G | [config](./segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.log.json)  |
| InternImage-XL | 512x1024   | 85.41 / 85.93  | 0.43s / iter       |  19.5h      | 330M    | 2364G | [config](./segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.log.json) |
