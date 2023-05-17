<div id="top" align="center">

# InternImage-based Baseline for Online HD Map Construction Challenge For Autonomous Driving
 </div>



If you need detaild information about the challenge, please refer to https://github.com/Tsinghua-MARS-Lab/Online-HD-Map-Construction-CVPR2023/tree/master
#### 1. Requirements
```bash
python>=3.8
torch==1.11 # recommend
mmcv-full>=1.5.2
mmdet==2.28.1
mmsegmentation==0.29.1
timm
numpy==1.23.5
mmdet3d==1.0.0rc6 # recommend
```


### 2. Install DCNv3 for InternImage
```bash
cd projects/ops_dcnv3
bash make.sh # requires torch>=1.10
```

### 3. Train with InternImage-Small

```bash
bash tools/dist_train.sh src/configs/vectormapnet_intern.py ${NUM_GPUS}
```

Notes: InatenImage provides abundant pre-trained model weights that can be used!!!


### 4. Performance compared to baseline

model name|weight|$\mathrm{mAP}$ | $\mathrm{AP}_{pc}$ | $\mathrm{AP}_{div}$ | $\mathrm{AP}_{bound}$ | 
----|:----------:| :--: | :--: | :--: | :--: | 
vectormapnet_intern|[Checkpoint](https://github.com/OpenGVLab/InternImage/releases/download/track_model/vectormapnet_internimage.pth) | 49.35 | 45.05 | 56.78 | 46.22 | 
vectormapnet_base|[Google Drive](https://drive.google.com/file/d/16D1CMinwA8PG1sd9PV9_WtHzcBohvO-D/view) | 42.79 | 37.22 | 50.47	 | 40.68 | 





## Citation

The evaluation metrics of this challenge follows [HDMapNet](https://arxiv.org/abs/2107.06307). We provide [VectorMapNet](https://arxiv.org/abs/2206.08920) as the baseline. Please cite:

```
@article{li2021hdmapnet,
    title={HDMapNet: An Online HD Map Construction and Evaluation Framework},
    author={Qi Li and Yue Wang and Yilun Wang and Hang Zhao},
    journal={arXiv preprint arXiv:2107.06307},
    year={2021}
}
```

Our dataset is built on top of the [Argoverse 2](https://www.argoverse.org/av2.html) dataset. Please also cite:

```
@INPROCEEDINGS {Argoverse2,
  author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
  title = {Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```


## License

Before participating in our challenge, you should register on the website and agree to the terms of use of the [Argoverse 2](https://www.argoverse.org/av2.html) dataset.
All code in this project is released under [GNU General Public License v3.0](./LICENSE).
