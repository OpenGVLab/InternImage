# Data

## Download

The files mentioned below can also be downloaded via [OpenDataLab](https://opendatalab.com/OpenLane-V2/download).
It is recommended to use provided [command line interface](https://opendatalab.com/OpenLane-V2/cli) for acceleration.

| Subset | Split | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Yun <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | md5 | Size |
| --- | --- | --- | --- | --- | --- |
| subset_A | sample |[sample](https://drive.google.com/file/d/1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6/view?usp=share_link) | [sample](https://pan.baidu.com/s/1ncqwDtuihKTBZROL5vdCAQ?pwd=psev) | 21c607fa5a1930275b7f1409b25042a0 | ~300M |
| subset_A | all | [info](https://drive.google.com/file/d/1t47lNF4H3WhSsAqgsl9lSLIeO0p6n8p4/view?usp=share_link) | [info](https://pan.baidu.com/s/1uXpX4hqlMJLm0W6l12dJ-A?pwd=6rzj) |95bf28ccf22583d20434d75800be065d | ~8.8G |
|  | train | [image_0](https://drive.google.com/file/d/1jio4Gj3dNlXmSzebO6D7Uy5oz4EaTNTq/view?usp=share_link) | [image_0](https://pan.baidu.com/s/12aV4CoT8znEY12q4M8XFiw?pwd=m204) | 8ade7daeec1b64f8ab91a50c81d812f6 | ~14.0G |
|  |  | [image_1](https://drive.google.com/file/d/1IgnvZ2UljL49AzNV6CGNGFLQo6tjNFJq/view?usp=share_link) | [image_1](https://pan.baidu.com/s/1SArnlA2_Om9o0xcGd6-EwA?pwd=khx8) | c78e776f79e2394d2d5d95b7b5985e0f | ~14.3G |
|  |  | [image_2](https://drive.google.com/file/d/1ViEsK5hukjMGfOm_HrCiQPkGArWrT91o/view?usp=share_link) | [image_2](https://pan.baidu.com/s/1ZghG7gwJqFrGxCEcUffp8A?pwd=0xgm) | 4bf09079144aa54cb4dcd5ff6e00cf79 | ~14.2G |
|  |  | [image_3](https://drive.google.com/file/d/1r3NYauV0JIghSmEihTxto0MMoyoh4waK/view?usp=share_link) | [image_3](https://pan.baidu.com/s/1ogwmXwS9u-B9nhtHlBTz5g?pwd=sqeg) | fd9e64345445975f462213b209632aee | ~14.4G |
|  |  | [image_4](https://drive.google.com/file/d/1aBe5yxNBew11YRRu-srQNwc5OloyKP4r/view?usp=share_link) | [image_4](https://pan.baidu.com/s/1tMAmUcZH2SzCiJoxwgk87w?pwd=i1au) | ae07e48c88ea2c3f6afbdf5ff71e9821 | ~14.5G |
|  |  | [image_5](https://drive.google.com/file/d/1Or-Nmsq4SU24KNe-cn9twVYVprYPUd_y/view?usp=share_link) | [image_5](https://pan.baidu.com/s/1sRyrhcSz-izW2U5x3UACSA?pwd=nzxx) | df62c1f6e6b3fb2a2a0868c78ab19c92 | ~14.2G |
|  |  | [image_6](https://drive.google.com/file/d/1mSWU-2nMzCO5PGF7yF9scoPntWl7ItfZ/view?usp=share_link) | [image_6](https://pan.baidu.com/s/1P3zn_L6EIGUHb43qWOJYWg?pwd=4wei) | 7bff1ce30329235f8e0f25f6f6653b8f | ~14.4G |
|  | val | [image_7](https://drive.google.com/file/d/19N5q-zbjE2QWngAT9xfqgOR3DROTAln0/view?usp=share_link) | [image_7](https://pan.baidu.com/s/1rRkPWg-zG2ygsbMhwXjPKg?pwd=qsvb) | c73af4a7aef2692b96e4e00795120504 | ~21.0G |
|  | test | [image_8](https://drive.google.com/file/d/1CvT9w0q8vPldfaajI5YsAqM0ZINT1vJv/view?usp=share_link) | [image_8](https://pan.baidu.com/s/10zjKeuAw350fwTYAeuSLxg?pwd=99ch) | fb2f61e7309e0b48e2697e085a66a259 | ~21.2G |
| subset_B | coming soon | - | - | - | - |

For files in Google Drive, you can use the following command by replacing `[FILE_ID]` and `[FILE_NAME]` accordingly:
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=[FILE_ID]' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=[FILE_ID]" -O [FILE_NAME]
```

## Preprocess
The dataset is preprocessed into pickle files representing different collections, which then be used for training models or evaluation:

```sh
cd data
python OpenLane-V2/preprocess.py 
```

## Hierarchy
The hierarchy of folder `OpenLane-V2/` is described below:
```
└── OpenLane-V2
    ├── train
    |   ├── [segment_id]
    |   |   ├── image
    |   |   |   ├── [camera]
    |   |   |   |   ├── [timestamp].jpg
    |   |   |   |   └── ...
    |   |   |   └── ...
    |   |   └── info
    |   |       ├── [timestamp].json
    |   |       └── ...
    |   └── ...
    ├── val
    |   └── ...
    ├── test
    |   └── ...
    ├── data_dict_example.json
    ├── data_dict_subset_A.json
    ├── data_dict_subset_B.json
    ├── openlanev2.md5
    └── preprocess.py

```

- `[segment_id]` specifies a sequence of frames, and `[timestamp]` specifies a single frame in a sequence.
- `image/` contains images captured by various cameras, and `info/` contains meta data and annotations of a single frame.
- `data_dict_[xxx].json` notes the split of train / val / test under the subset of data.

## Meta Data
The json files under the `info/` folder contain meta data and annotations for each frame.
Each file is formatted as follows:

```
{
    'version':                              <str> -- version
    'segment_id':                           <str> -- segment_id
    'meta_data': {
        'source':                           <str> -- name of the original dataset
        'source_id':                        <str> -- original identifier of the segment
    }
    'timestamp':                            <int> -- timestamp of the frame
    'sensor': {
        [camera]: {                         <str> -- name of the camera
            'image_path':                   <str> -- image path
            'extrinsic':                    <dict> -- extrinsic parameters of the camera
            'intrinsic':                    <dict> -- intrinsic parameters of the camera
        },
        ...
    }                              
    'pose':                                 <dict> -- ego pose
    'annotation':                           <dict> -- anntations for the current frame
}
```

## Annotations
For a single frame, annotations are formatted as follow:

```
{
    'lane_centerline': [                    (n lane centerlines in the current frame)
        {   
            'id':                           <int> -- unique ID in the current frame
            'points':                       <float> [n, 3] -- 3D coordiate
            'confidence':                   <float> -- confidence, only for prediction
        },
        ...
    ],
    'traffic_element': [                    (k traffic elements in the current frame)
        {   
            'id':                           <int> -- unique ID in the current frame
            'category':                     <int> -- traffic element category
                                                1: 'traffic_light',
                                                2: 'road_sign',
            'attribute':                    <int> -- attribute of traffic element
                                                0:  'unknown',
                                                1:  'red',
                                                2:  'green',
                                                3:  'yellow',
                                                4:  'go_straight',
                                                5:  'turn_left',
                                                6:  'turn_right',
                                                7:  'no_left_turn',
                                                8:  'no_right_turn',
                                                9:  'u_turn',
                                                10: 'no_u_turn',
                                                11: 'slight_left',
                                                12: 'slight_right',
            'points':                       <float> [2, 2] -- top-left and bottom-right corners of the 2D bounding box
            'confidence':                   <float> -- confidence, only for prediction
        },
        ...
    ],
    'topology_lclc':                        <float> [n, n] -- adjacent matrix among lane centerlines
    'topology_lcte':                        <float> [n, k] -- adjacent matrix between lane centerlines and traffic elements
}
```

- `id` is the identifier of a lane centerline or traffic element and is consistent in a sequence.
For predictions, it can be randomly assigned but unique in a single frame.
- `topology_lclc` and `topology_lcte` are adjacent matrices, where row and column are sorted according to the order of the lists `lane_centerline` and `traffic_element`. 
It is a MUST to keep the ordering the same for correct evaluation. 
For ground truth, only 0 or 1 is a valid boolean value for an element in the matrix. 
For predictions, the value varies from 0 to 1, representing the confidence of the predicted relationship. 
- #lane_centerline and #traffic_element are not required to be equal between ground truth and predictions. 
In the process of evaluation, a matching of ground truth and predictions is determined.
