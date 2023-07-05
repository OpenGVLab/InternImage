# Copyright (c) OpenMMLab. All rights reserved.
from .mapillary import MapillaryDataset  # noqa: F401,F403
from .nyu_depth_v2 import NYUDepthV2Dataset  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
from .dataset_wrappers import ConcatDataset


__all__ = [
    'MapillaryDataset', 'NYUDepthV2Dataset', 'ConcatDataset'
]