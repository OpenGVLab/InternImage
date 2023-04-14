# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.detr import DETR


@DETECTORS.register_module()
class DINO(DETR):
    
    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)