# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# collect.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
from tqdm import tqdm

from ..io import io


def collect(root_path : str, data_dict : dict, collection : str, point_interval : int = 1) -> None:
    r"""
    Load meta data of data in data_dict,
    and store in a .pkl with split as file name.

    Parameters
    ----------
    root_path : str
    data_dict : dict
        A dict contains ids of data to be preprocessed.
    collection : str
        Name of the collection.
    point_interval : int
        Interval for subsampling points of lane centerlines,
        not subsampling as default.

    """
    data_list = [(split, segment_id, timestamp.split('.')[0]) \
        for split, segment_ids in data_dict.items() \
            for segment_id, timestamps in segment_ids.items() \
                for timestamp in timestamps
    ]
    meta = {
        (split, segment_id, timestamp): io.json_load(f'{root_path}/{split}/{segment_id}/info/{timestamp}.json') \
            for split, segment_id, timestamp in tqdm(data_list, desc=f'collecting {collection}', ncols=80)
    }

    for identifier, frame in meta.items():
        for k, v in meta[identifier]['pose'].items():
            meta[identifier]['pose'][k] = np.array(v, dtype=np.float64)
        for camera in meta[identifier]['sensor'].keys():
            for para in ['intrinsic', 'extrinsic']:
                for k, v in meta[identifier]['sensor'][camera][para].items():
                    meta[identifier]['sensor'][camera][para][k] = np.array(v, dtype=np.float64)

        if 'annotation' not in frame:
            continue
        for i, lane_centerline in enumerate(frame['annotation']['lane_centerline']):
            meta[identifier]['annotation']['lane_centerline'][i]['points'] = np.array(lane_centerline['points'][::point_interval], dtype=np.float32)
        for i, traffic_element in enumerate(frame['annotation']['traffic_element']):
            meta[identifier]['annotation']['traffic_element'][i]['points'] = np.array(traffic_element['points'], dtype=np.float32)
        meta[identifier]['annotation']['topology_lclc'] = np.array(meta[identifier]['annotation']['topology_lclc'], dtype=np.int8)
        meta[identifier]['annotation']['topology_lcte'] = np.array(meta[identifier]['annotation']['topology_lcte'], dtype=np.int8)

    io.pickle_dump(f'{root_path}/{collection}.pkl', meta)
