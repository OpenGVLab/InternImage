# ADE20K

Introduced by Zhou et al. in [Scene Parsing Through ADE20K Dataset](https://paperswithcode.com/paper/scene-parsing-through-ade20k-dataset).

The ADE20K semantic segmentation dataset contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.


## Model Zoo

### UperNet + InternImage


| backbone       | resolution | mIoU (ss/ms) | train speed | train time | #param | FLOPs | Config | Download            |
|:--------------:|:----------:|:-----------:|:-----------:|:----------:|:-------:|:-----:|:-----:|:-------------------:|
| InternImage-T  | 512x512    | 47.9 / 48.1  | 0.23s / iter       | 10.5h      | 59M     | 944G  | [config](./upernet_internimage_t_512_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_t_512_160k_ade20k.log.json)   | 
| InternImage-S  | 512x512    | 50.1 / 50.9  | 0.25s / iter       | 11.5h      | 80M     | 1017G | [config](./upernet_internimage_s_512_160k_ade20k.py)  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_s_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_s_512_160k_ade20k.log.json)  | 
| InternImage-B  | 512x512    | 50.8 / 51.3  | 0.26s / iter       | 12h        | 128M    | 1185G | [config](./upernet_internimage_b_512_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_b_512_160k_ade20k.log.json)  | 
| InternImage-L  | 640x640    | 53.9 / 54.1  | 0.42s / iter       | 19h        | 256M    | 2526G | [config](./upernet_internimage_l_640_160k_ade20k.py)| [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_640_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_l_640_160k_ade20k.log.json)  | 
| InternImage-XL | 640x640    | 55.0 / 55.3  | 0.47s / iter       | 22h        | 368M    | 3142G | [config](./upernet_internimage_xl_640_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_640_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_xl_640_160k_ade20k.log.json) | 
| InternImage-H  | 896x896    | 59.9 / 60.3  | 0.94s / iter       | 2d (2n)       | 1.12B    | 3566G | [config](./upernet_internimage_h_896_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/upernet_internimage_h_896_160k_ade20k.log.json) | 

- Training speed is measured with A100 GPU.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.
- The logs are our recent newly trained ones. There are slight differences between the results in logs and our paper.


### Mask2Former + InternImage

| backbone       | resolution | mIoU (ss/ms) | train speed | train time | #param | FLOPs | Config | Download            |
|:--------------:|:----------:|:-----------:|:-----------:|:----------:|:-------:|:-----:|:-----:|:-------------------:|
| InternImage-H  | 896x896    | 62.6 / 62.9  | 1.21s / iter       | 1.5d (2n)       | 1.31B    | 4635G | [config](./mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.log.json) |
