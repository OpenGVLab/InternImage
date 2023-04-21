# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# match_costs.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

import torch

from mmdet.core.bbox.match_costs.builder import MATCH_COST


@MATCH_COST.register_module()
class LaneL1Cost:
    r"""
    Notes
    -----
    Adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/match_costs/match_cost.py#L11.

    """
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lane_pred, gt_lanes):
        lane_cost = torch.cdist(lane_pred, gt_lanes, p=1)
        return lane_cost * self.weight
