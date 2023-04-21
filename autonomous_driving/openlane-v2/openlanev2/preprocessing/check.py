# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# check.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
from iso3166 import countries
from functools import reduce


def check_results(results : dict) -> None:
    r"""
    Check format of results.

    Parameters
    ----------
    results : dcit
        Dict storing predicted results.

    """
    valid = True

    if not isinstance(results, dict):
        raise Exception(f'Type of result should be dict')

    for key in ['method', 'e-mail', 'institution / company', 'country / region']:
        if key in results:
            if not isinstance(results[key], str):
                raise Exception(f'Type of value in key [{key}] should be str')
            if key == 'country / region':
                try:
                    countries.get(results[key])
                except Exception:
                    raise Exception(f'Please specify a valid [{key}] according to ISO3166')
        else:
            valid = False
            print(f'\n*** Missing key [{key}] for a valid submission ***\n')

    for key in ['authors']:
        if key in results:
            if not isinstance(results[key], list):
                raise Exception(f'Type of value in key [{key}] should be list')
            if len(results[key]) > 10:
                raise Exception(f'The number of authors should not exceed 10')
        else:
            valid = False
            print(f'\n*** Missing key [{key}] for a valid submission ***\n')

    for key in ['results']:
        if key not in results:
            raise Exception(f'Miss key [{key}].')
        if not isinstance(results[key], dict):
            raise Exception(f'Type of value in key [{key}] should be dict')

    for token, predictions in results['results'].items():
        if not isinstance(predictions, dict):
            raise Exception(f'Type of value in key [results/{token}] should be dict')
        predictions = predictions['predictions']
        if not isinstance(predictions, dict):
            raise Exception(f'Type of value in key [results/{token}/predictions] should be dict')

        ids = {}
        for key in ['lane_centerline', 'traffic_element']:
            if key not in predictions:
                raise Exception(f'Miss key [results/{token}/predictions/{key}].')
            if not isinstance(predictions[key], list):
                raise Exception(f'Type of value in key [results/{token}/predictions/{key}] should be list')

            for instance in predictions[key]:
                for k in ['id', 'points', 'confidence']:
                    if k not in instance:
                        raise Exception(f'Miss key [results/{token}/predictions/{key}/k].')
                if key == 'traffic_element':
                    if 'attribute' not in instance:
                        raise Exception(f'Miss key [results/{token}/predictions/{key}/k].')

                points = instance['points']
                if not isinstance(points, np.ndarray):
                    raise Exception(f'Type of value in key [results/{token}/predictions/{key}/{instance["id"]}] should be np.ndarray')
                points = np.array(points)
                if key == 'lane_centerline' and not (points.ndim == 2 and points.shape[1] == 3):
                    raise Exception(f'Shape of points in instance [results/{token}/predictions/{key}/{instance["id"]}] should be (#points, 3) but not {points.shape}')
                if key == 'traffic_element' and not (points.ndim == 2 and points.shape == (2, 2)):
                    raise Exception(f'Shape of points in instance [results/{token}/predictions/{key}/{instance["id"]}] should be (2, 2) but not {points.shape}')
                
            ids[key] = [instance['id'] for instance in predictions[key]]

        ids_check = reduce(lambda x, y: x + y, ids.values(), [])
        if len(set(ids_check)) != len(ids_check):
            raise Exception(f'IDs are not unique in [results/{token}/predictions]')
                
        if 'topology_lclc' not in predictions:
            raise Exception(f'Miss key [results/{token}/predictions/topology_lclc].')
        topology_lclc = predictions['topology_lclc']
        if not isinstance(topology_lclc, np.ndarray):
            raise Exception(f'Type of value in key [results/{token}/predictions/topology_lclc] should be np.ndarray')
        topology_lclc = np.array(topology_lclc)
        if not (topology_lclc.ndim == 2 and topology_lclc.shape[0] == len(ids['lane_centerline']) and topology_lclc.shape[1] == len(ids['lane_centerline'])):
            raise Exception(f'Shape of adjacent matrix of [results/{token}/predictions/topology_lclc] should be (#lane_centerline, #lane_centerline) but not {topology_lclc.shape}')
        
        if 'topology_lcte' not in predictions:
            raise Exception(f'Miss key [results/{token}/predictions/topology_lcte].')
        topology_lcte = predictions['topology_lcte']
        if not isinstance(topology_lcte, np.ndarray):
            raise Exception(f'Type of value in key [results/{token}/predictions/topology_lcte] should be np.ndarray')
        topology_lcte = np.array(topology_lcte)
        if not (topology_lcte.ndim == 2 and topology_lcte.shape[0] == len(ids['lane_centerline']) and topology_lcte.shape[1] == len(ids['traffic_element'])):
            raise Exception(f'Shape of adjacent matrix of [results/{token}/predictions/topology_lcte] should be (#lane_centerline, #traffic_element) but not {topology_lcte.shape}')

    return valid
