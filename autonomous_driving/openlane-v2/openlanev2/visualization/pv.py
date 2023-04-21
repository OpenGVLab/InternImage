# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# pv.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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


def _draw_traffic_element(image, traffic_element):
    top_left = (
        int(traffic_element['points'][0][0]),
        int(traffic_element['points'][0][1]),
    )
    bottom_right = (
        int(traffic_element['points'][1][0]),
        int(traffic_element['points'][1][1]),
    )
    
    color = COLOR_DICT[traffic_element['attribute']]

    cv2.rectangle(image, top_left, bottom_right, color=color, thickness=THICKNESS, lineType=cv2.LINE_AA)
    
def _project(points, intrinsic, extrinsic):
    if points is None:
        return points
    
    points_in_cam_cor = np.linalg.pinv(np.array(extrinsic['rotation'])) \
        @ (points.T - np.array(extrinsic['translation']).reshape(3, -1))
    points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0]

    if points_in_cam_cor.shape[1] > 1:
        points_on_image_cor = np.array(intrinsic['K']) @ points_in_cam_cor
        points_on_image_cor = points_on_image_cor / (points_on_image_cor[-1, :].reshape(1, -1))
        points_on_image_cor = points_on_image_cor[:2, :].T
    else:
        points_on_image_cor = None
    
    return points_on_image_cor

def _draw_lane_centerline(image, lane_centerline, intrinsic, extrinsic, with_attribute):
    points = _project(interp_arc(lane_centerline['points']), intrinsic, extrinsic)
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

            try:
                cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=THICKNESS, lineType=cv2.LINE_AA)
            except Exception:
                return

def _draw_topology(image, topology, intrinsic, extrinsic):
    coord_from = [
        (topology['traffic_element'][0][0] + topology['traffic_element'][0][0]) / 2,
        topology['traffic_element'][1][1],
    ]
    
    points = _project(interp_arc(topology['lane_centerline']), intrinsic, extrinsic)
    if points is None:
        return
    coord_to = points[len(points) // 2]

    color = COLOR_DICT[topology['attribute']]
    
    mid = ((coord_to[0] + coord_from[0]) / 2, (coord_to[1] + coord_from[1]) / 2 - 50,)
    curve = np.array([coord_from, mid, coord_to])
    pts_fit = np.polyfit(curve[:, 0], curve[:, 1], 2)
    xs = np.linspace(curve[0][0], curve[-1][0], 1000)
    ys = pts_fit[0] * xs**2 + pts_fit[1] * xs + pts_fit[2]
    curve = np.int_([np.array([np.transpose(np.vstack([xs, ys]))])])

    cv2.polylines(image, curve, isClosed=False, color=color, thickness=THICKNESS//3, lineType=cv2.LINE_AA)

def draw_annotation_pv(camera, image, annotation, intrinsic, extrinsic, with_attribute, with_topology):
    for lane_centerline in annotation['lane_centerline']:
        _draw_lane_centerline(image, lane_centerline, intrinsic, extrinsic, with_attribute)
    if camera in ['ring_front_center', 'CAM_FRONT']:
        for traffic_element in annotation['traffic_element']:
            _draw_traffic_element(image, traffic_element)
        if with_topology:
            for topology in annotation['topology']:
                _draw_topology(image, topology, intrinsic, extrinsic)
    return image
