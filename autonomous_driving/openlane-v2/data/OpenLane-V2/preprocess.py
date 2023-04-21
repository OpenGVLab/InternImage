# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# preprocess.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
# =============================================================================='

from openlanev2.io import io
from openlanev2.preprocessing import collect

root_path = './OpenLane-V2'
for file in io.os_listdir(root_path):
    if file.endswith('json'):
        subset = file.split('.')[0]
        for split, segments in io.json_load(f'{root_path}/{file}').items():
            point_interval = 1 if split == 'train' else 20
            collect(root_path, {split: segments}, f'{subset}_{split}', point_interval=point_interval)
