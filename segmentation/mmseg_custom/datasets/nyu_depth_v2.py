# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class NYUDepthV2Dataset(CustomDataset):
    """NYU Depth V2 dataset.
    """

    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair',
               'sofa', 'table', 'door', 'window', 'bookshelf',
               'picture', 'counter', 'blinds', 'desk', 'shelves',
               'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
               'clothes', 'ceiling', 'books', 'refridgerator', 'television',
               'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
               'person', 'night stand', 'toilet', 'sink', 'lamp',
               'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')

    
    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],]

    def __init__(self, split, **kwargs):
        super(NYUDepthV2Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=True,
            **kwargs)
        