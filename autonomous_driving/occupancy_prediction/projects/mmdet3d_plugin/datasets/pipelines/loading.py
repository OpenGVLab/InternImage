import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os

@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        # print(results.keys())
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(self.data_root,occ_gt_path)

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)