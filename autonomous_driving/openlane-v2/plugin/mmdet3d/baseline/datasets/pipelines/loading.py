# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# loading.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

import mmcv
from mmdet.datasets import PIPELINES
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):

    def __call__(self, results):
        filename = results['img_paths']
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in results['img']]
        return results
