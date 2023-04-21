# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# formating.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from mmcv.parallel import DataContainer as DC
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class CustomDefaultFormatBundle:

    def __init__(self):
        pass

    def __call__(self, results):

        temp = to_tensor(np.concatenate([i[None, ...] for i in results['img']], axis=0))
        results['img'] = DC(temp.permute(0, 3, 1, 2), stack=True)
        
        if 'gt_lc' in results:
            results['gt_lc'] = DC(to_tensor(results['gt_lc']))
        if 'gt_lc_labels' in results:
            results['gt_lc_labels'] = DC(to_tensor(results['gt_lc_labels']))
        if 'gt_te' in results:
            results['gt_te'] = DC(to_tensor(results['gt_te']))
        if 'gt_te_labels' in results:
            results['gt_te_labels'] = DC(to_tensor(results['gt_te_labels']))
        if 'gt_topology_lclc' in results:
            results['gt_topology_lclc'] = DC(to_tensor(results['gt_topology_lclc']))
        if 'gt_topology_lcte' in results:
            results['gt_topology_lcte'] = DC(to_tensor(results['gt_topology_lcte']))
        
        return results
