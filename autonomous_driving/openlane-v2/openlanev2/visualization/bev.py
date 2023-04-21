# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# bev.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

import cv2
import numpy as np

from .utils import THICKNESS, COLOR_DEFAULT, COLOR_DICT, interp_arc


BEV_SCALE = 10
BEV_RANGE = [-50, 50, -25, 25]


def _draw_lane_centerline(image, lane_centerline, with_attribute):
    points = np.array(lane_centerline['points'])
    points = BEV_SCALE * (-points[:, :2] + np.array([BEV_RANGE[1] , BEV_RANGE[3]]))
    points = interp_arc(points)
    if points is None:
        return
    
    if with_attribute and len(set(lane_centerline['attributes']) - set([0])):
        colors = [COLOR_DICT[a] for a in set(lane_centerline['attributes']) - set([0])]
    else:
        colors = [COLOR_DEFAULT]
    
    for idx, color in enumerate(colors):
        for i in range(len(points) - 1):
            x1 = int(points[i][0] + idx * THICKNESS * 1.5)
            y1 = int(points[i][1] + idx * THICKNESS * 1.5)
            x2 = int(points[i+1][0] + idx * THICKNESS * 1.5)
            y2 = int(points[i+1][1] + idx * THICKNESS * 1.5)

            cv2.line(image, pt1=(y1, x1), pt2=(y2, x2), color=color, thickness=THICKNESS, lineType=cv2.LINE_AA)
        
def _draw_vertex(image, lane_centerline):
    points = BEV_SCALE * (-np.array(lane_centerline['points'])[:, :2] + np.array([BEV_RANGE[1] , BEV_RANGE[3]]))    
    
    cv2.circle(image, (int(points[0, 1]), int(points[0, 0])), int(THICKNESS * 1.5), COLOR_DEFAULT, -1)
    cv2.circle(image, (int(points[-1, 1]), int(points[-1, 0])), int(THICKNESS * 1.5), COLOR_DEFAULT, -1)

def draw_annotation_bev(annotation, with_attribute):
    image = np.ones((
        BEV_SCALE * (BEV_RANGE[1] - BEV_RANGE[0]),
        BEV_SCALE * (BEV_RANGE[3] - BEV_RANGE[2]),
        3,
    ), dtype=np.int32) * 191
    for lane_centerline in annotation['lane_centerline']:
        _draw_lane_centerline(image, lane_centerline, with_attribute)
    for lane_centerline in annotation['lane_centerline']:
        _draw_vertex(image, lane_centerline)
    return image
