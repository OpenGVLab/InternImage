# UPerNet

[Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/pdf/1807.10221.pdf)

## Introduction

<!-- [ABSTRACT] -->

Humans recognize the visual world at multiple levels: we effortlessly categorize scenes and detect objects inside, while also identifying the textures and surfaces of the objects along with their different compositional parts. In this paper, we study a new task called Unified Perceptual Parsing, which requires the machine vision systems to recognize as many visual concepts as possible from a given image. A multi-task framework called UPerNet and a training strategy are developed to learn from heterogeneous image annotations. We benchmark our framework on Unified Perceptual Parsing and show that it is able to effectively segment a wide range of concepts from images. The trained networks are further applied to discover visual knowledge in natural scenes. Models are available at [this https URL](https://github.com/CSAILVision/unifiedparsing).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142903077-44e8e0da-7276-4bda-bd2b-0df1680ca845.png" width="70%"/>
</div>

## Results and models

### ADE20K

**ADE20K Semantic Segmentation**

|    backbone    | resolution | single scale | multi scale | #params | FLOPs | Download | 
| :------------: | :--------: | :----------: | :---------: | :-----: | :---: |   :---:  |
| InternImage-T  |  512x512   |     47.9     |    48.1     |   59M   | 944G  | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_t_512_160k_ade20k.pth) \| [cfg](./upernet_internimage_t_512_160k_ade20k.py) |
| InternImage-S  |  512x512   |     50.1     |    50.9     |   80M   | 1017G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_s_512_160k_ade20k.pth) \| [cfg](./upernet_internimage_s_512_160k_ade20k.py) |
| InternImage-B  |  512x512   |     50.8     |    51.3     |  128M   | 1185G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_b_512_160k_ade20k.pth) \| [cfg](./upernet_internimage_b_512_160k_ade20k.py) |
| InternImage-L  |  640x640   |     53.9     |    54.1     |  256M   | 2526G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_l_640_160k_ade20k.pth) \| [cfg](./upernet_internimage_l_640_160k_ade20k.py) |
| InternImage-XL |  640x640   |     55.0     |    55.3     |  368M   | 3142G | [ckpt](https://github.com/OpenGVLab/InternImage/releases/download/seg_models/upernet_internimage_xl_640_160k_ade20k.pth) \| [cfg](./upernet_internimage_xl_640_160k_ade20k.py) |

