---
license: mit
pipeline_tag: image-classification
library_name: transformers
tags:
  - internimage
  - custom_code
datasets:
  - ILSVRC/imagenet-1k
---

# InternImage Model Card

## Introduction

InternImage is an advanced vision foundation model developed by researchers from Shanghai AI Laboratory, Tsinghua University, and other institutions. Unlike models based on Transformers, InternImage employs DCNv3 as its core operator. This approach equips the model with dynamic and effective receptive fields required for downstream tasks like object detection and segmentation, while enabling adaptive spatial aggregation.

<div style="text-align: center;"> <img src="https://github.com/OpenGVLab/InternImage/raw/master/docs/figs/arch.png" style="width:60%;" /> </div>

## Performance

- InternImage achieved an impressive Top-1 accuracy of 90.1% on the ImageNet benchmark dataset using only publicly available data for image classification. Apart from two undisclosed models trained with additional datasets by Google and Microsoft, InternImage is the only open-source model that achieves a Top-1 accuracy of over 90.0%, and it is also the largest model in scale worldwide.
- InternImage outperformed all other models worldwide on the COCO object detection benchmark dataset with a remarkable mAP of 65.5, making it the only model that surpasses 65 mAP in the world.
- InternImage also demonstrated world's best performance on 16 other important visual benchmark datasets, covering a wide range of tasks such as classification, detection, and segmentation, making it the top-performing model across multiple domains.

## Released Models

### Open‑Source Visual Pretrained Models

|                                       huggingface name                                        |  model  name   |       pretrain       | resolution | #param |
| :-------------------------------------------------------------------------------------------: | :------------: | :------------------: | :--------: | :----: |
|        [internimage_l_22k_384](https://huggingface.co/OpenGVLab/internimage_l_22k_384)        | InternImage-L  |        IN-22K        |  384x384   |  223M  |
|       [internimage_xl_22k_384](https://huggingface.co/OpenGVLab/internimage_xl_22k_384)       | InternImage-XL |        IN-22K        |  384x384   |  335M  |
| [internimage_h_jointto22k_384](https://huggingface.co/OpenGVLab/internimage_h_jointto22k_384) | InternImage-H  | Joint 427M -> IN-22K |  384x384   | 1.08B  |
| [internimage_g_jointto22k_384](https://huggingface.co/OpenGVLab/internimage_g_jointto22k_384) | InternImage-G  | Joint 427M -> IN-22K |  384x384   |   3B   |

### ImageNet-1K Image Classification

|                                     huggingface name                                      |   model name   |       pretrain       | resolution | acc@1 | #param | FLOPs |
| :---------------------------------------------------------------------------------------: | :------------: | :------------------: | :--------: | :---: | :----: | :---: |
|       [internimage_t_1k_224](https://huggingface.co/OpenGVLab/internimage_t_1k_224)       | InternImage-T  |        IN-1K         |  224x224   | 83.5  |  30M   |  5G   |
|       [internimage_s_1k_224](https://huggingface.co/OpenGVLab/internimage_s_1k_224)       | InternImage-S  |        IN-1K         |  224x224   | 84.2  |  50M   |  8G   |
|       [internimage_b_1k_224](https://huggingface.co/OpenGVLab/internimage_b_1k_224)       | InternImage-B  |        IN-1K         |  224x224   | 84.9  |  97M   |  16G  |
|  [internimage_l_22kto1k_384](https://huggingface.co/OpenGVLab/internimage_l_22kto1k_384)  | InternImage-L  |        IN-22K        |  384x384   | 87.7  |  223M  | 108G  |
| [internimage_xl_22kto1k_384](https://huggingface.co/OpenGVLab/internimage_xl_22kto1k_384) | InternImage-XL |        IN-22K        |  384x384   | 88.0  |  335M  | 163G  |
|  [internimage_h_22kto1k_640](https://huggingface.co/OpenGVLab/internimage_h_22kto1k_640)  | InternImage-H  | Joint 427M -> IN-22K |  640x640   | 89.6  | 1.08B  | 1478G |
|  [internimage_g_22kto1k_512](https://huggingface.co/OpenGVLab/internimage_g_22kto1k_512)  | InternImage-G  | Joint 427M -> IN-22K |  512x512   | 90.1  |   3B   | 2700G |

## DCNv3 CUDA Kernel Installation

If you do not install the CUDA version of DCNv3, InternImage will automatically fall back to a PyTorch implementation. However, the CUDA implementation can significantly reduce GPU memory usage and improve inference efficiency.

**Installation Tutorial:**

1. Open your terminal and run:

   ```bash
   git clone https://github.com/OpenGVLab/InternImage.git
   cd InternImage/classification/ops_dcnv3
   ```

2. Make sure you have an available GPU for compilation, then run:

   ```bash
   sh make.sh
   ```

This will compile the CUDA version of DCNv3. Once installed, InternImage will automatically leverage the optimized CUDA implementation for better performance.

## Usage with Transformers

Below are two usage examples for InternImage with the Transformers framework:

### Example 1: Using InternImage as an Image Backbone

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

# Replace 'model_name' with the appropriate model identifier
model_name = "OpenGVLab/internimage_t_1k_224"  # example model

# Prepare the image
image_path = 'img.png'
image_processor = CLIPImageProcessor.from_pretrained(model_name)
image = Image.open(image_path)
image = image_processor(images=image, return_tensors='pt').pixel_values
print('image shape:', image.shape)

# Load the model as a backbone
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
# 'hidden_states' contains the outputs from the 4 stages of the InternImage backbone
hidden_states = model(image).hidden_states
```

### Example 2: Using InternImage for Image Classification

```python
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, CLIPImageProcessor

# Replace 'model_name' with the appropriate model identifier
model_name = "OpenGVLab/internimage_t_1k_224"  # example model

# Prepare the image
image_path = 'img.png'
image_processor = CLIPImageProcessor.from_pretrained(model_name)
image = Image.open(image_path)
image = image_processor(images=image, return_tensors='pt').pixel_values
print('image shape:', image.shape)

# Load the model as an image classifier
model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
logits = model(image).logits
label = torch.argmax(logits, dim=1)
print("Predicted label:", label.item())
```

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```Bibtex
@inproceedings{wang2023internimage,
  title={Internimage: Exploring large-scale vision foundation models with deformable convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14408--14419},
  year={2023}
}
```
