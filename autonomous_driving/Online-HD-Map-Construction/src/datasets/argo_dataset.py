from .base_dataset import BaseMapDataset
from mmdet.datasets import DATASETS
import numpy as np
from time import time
import mmcv
import os
from shapely.geometry import LineString

@DATASETS.register_module()
class AV2Dataset(BaseMapDataset):
    """Argoverse2 map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config,
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = []
        for seg_id, sequence in ann.items():
            samples.extend(sequence)
        samples = samples[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        
        if not self.test_mode:
            ann = sample['annotation']

            # collected required keys
            map_label2geom = {}
            for k, v in ann.items():
                if k in self.cat2id.keys():
                    map_label2geom[self.cat2id[k]] = [LineString(np.array(l)[:, :3]) for l in v]
        
        ego2img_rts = []
        cams = sample['sensor']
        for c in cams.values():
            extrinsic, intrinsic = np.array(
                c['extrinsic']), np.array(c['intrinsic'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        pose = sample['pose']
        input_dict = {
            'token': sample['timestamp'],
            'img_filenames': [os.path.join(self.root_path, c['image_path']) for c in cams.values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsic'] for c in cams.values()],
            # extrinsics are 4x4 tranform matrix, NOTE: **ego2cam**
            'cam_extrinsics': [c['extrinsic'] for c in cams.values()],
            'ego2img': ego2img_rts,
            'ego2global_translation': pose['ego2global_translation'], 
            'ego2global_rotation': pose['ego2global_rotation'],
        }
        if not self.test_mode:
            input_dict.update({'map_geoms': map_label2geom}) # {0: List[ped_crossing(LineString)], 1: ...}})

        return input_dict