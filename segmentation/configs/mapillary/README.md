
# Mapillary Vistas

Introduced by Neuhold et al. in [The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes](http://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf)

Mapillary Vistas Dataset is a diverse street-level imagery dataset with pixel‑accurate and instance‑specific human annotations for understanding street scenes around the world. 

We first pretrain our models on the Mapillary Vistas dataset, then finetune them on the Cityscapes dataset.


## Model Zoo

### UperNet + InternImage

| backbone       | resolution |  schd | train speed | train time | #params | FLOPs | Config | Download     |
|:--------------:|:----------:|:------------:|:-----------:|:-----------:|:-------:|:-----:|:------:|:------------:|
| InternImage-L  | 512x1024   | 80k  | 0.50s / iter | 11.5h    | 256M  | 3234G | [config](./upernet_internimage_l_512x1024_80k_mapillary.py)  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_512x1024_80k_mapillary.pth) |
| InternImage-XL | 512x1024   | 80k  | 0.56s / iter | 13h    | 368M  | 4022G | [config](./upernet_internimage_xl_512x1024_80k_mapillary.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_512x1024_80k_mapillary.pth) |

### SegFormerHead + InternImage

| backbone       | resolution |  schd | train speed | train time | #params | FLOPs | Config | Download |
|:--------------:|:----------:|:------------:|:-----------:|:-----------:|:-------:|:-----:|:-----:|:---------:|
| InternImage-L  | 512x1024   | 80k  | 0.37s / iter       |   9h       | 220M    | 1580G | [config](./segformer_internimage_l_512x1024_80k_mapillary.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_l_512x1024_80k_mapillary.pth)   |
| InternImage-XL | 512x1024   | 80k  | 0.43s / iter       |  10h      | 330M    | 2364G | [config](./segformer_internimage_xl_512x1024_80k_mapillary.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_xl_512x1024_80k_mapillary.pth)  |

