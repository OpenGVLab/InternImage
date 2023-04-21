# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# custom_ipm_view_transformer.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet3d.models import NECKS


def get_campos(reference_points, ego2cam, img_shape):
    '''
        Find the each refence point's corresponding pixel in each camera
        Args: 
            reference_points: [B, num_query, 3]
            ego2cam: (B, num_cam, 4, 4)
        Outs:
            reference_points_cam: (B*num_cam, num_query, 2)
            mask:  (B, num_cam, num_query)
            num_query == W*H
    '''

    ego2cam = reference_points.new_tensor(ego2cam)  # (B, N, 4, 4)
    reference_points = reference_points.clone()

    B, num_query = reference_points.shape[:2]
    num_cam = ego2cam.shape[1]

    # reference_points (B, num_queries, 4)
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = reference_points.view(
        B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)

    ego2cam = ego2cam.view(
        B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    # reference_points_cam (B, num_cam, num_queries, 4)
    reference_points_cam = (ego2cam @ reference_points).squeeze(-1)

    eps = 1e-9
    mask = (reference_points_cam[..., 2:3] > eps)

    reference_points_cam =\
        reference_points_cam[..., 0:2] / \
        reference_points_cam[..., 2:3] + eps

    reference_points_cam[..., 0] /= img_shape[1]
    reference_points_cam[..., 1] /= img_shape[0]

    # from 0~1 to -1~1
    reference_points_cam = (reference_points_cam - 0.5) * 2

    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))

    # (B, num_cam, num_query)
    mask = mask.view(B, num_cam, num_query)
    reference_points_cam = reference_points_cam.view(B*num_cam, num_query, 2)

    return reference_points_cam, mask

def construct_plane_grid(xbound, ybound, height: float, dtype=torch.float32):
    '''
        Returns:
            plane: H, W, 3
    '''

    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    x = torch.linspace(xmin, xmax, num_x, dtype=dtype)
    y = torch.linspace(ymin, ymax, num_y, dtype=dtype)

    # [num_y, num_x]
    y, x = torch.meshgrid(y, x)

    z = torch.ones_like(x) * height

    # [num_y, num_x, 3]
    plane = torch.stack([x, y, z], dim=-1)

    return plane

@NECKS.register_module()
class CustomIPMViewTransformer(BaseModule):
    r"""
    Notes
    -----
    Adapted from https://github.com/Mrmoore98/VectorMapNet_code/blob/mian/plugin/models/backbones/ipm_backbone.py#L238.

    """
    def __init__(self,         
                 num_cam,        
                 xbound,
                 ybound,
                 zbound,
                 out_channels,
                 ):
        super().__init__()
        self.x_bound = xbound
        self.y_bound = ybound
        heights = [zbound[0]+i*zbound[2] for i in range(int((zbound[1]-zbound[0])//zbound[2])+1)]
        self.heights = heights

        self.num_cam = num_cam

        self.outconvs =\
            nn.Conv2d((out_channels+3)*len(heights), out_channels, 
                        kernel_size=3, stride=1, padding=1)  # same

        # bev_plane
        bev_planes = [construct_plane_grid(
            xbound, ybound, h) for h in self.heights]
        self.register_buffer('bev_planes', torch.stack(
            bev_planes),)  # nlvl,bH,bW,2

    def forward(self, cam_feat, ego2cam, img_shape):
        '''
            inverse project 
            Args:
                cam_feat: B*ncam, C, cH, cW
                img_shape: tuple(H, W)
            Returns:
                project_feat: B, C, nlvl, bH, bW
                bev_feat_mask: B, 1, nlvl, bH, bW
        '''
        B = ego2cam.shape[0]
        C = cam_feat.shape[1]
        bev_grid = self.bev_planes.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        nlvl, bH, bW = bev_grid.shape[1:4]
        bev_grid = bev_grid.flatten(1, 3)  # B, nlvl*W*H, 3

        # Find points in cam coords
        # bev_grid_pos: B*ncam, nlvl*bH*bW, 2
        bev_grid_pos, bev_cam_mask = get_campos(bev_grid, ego2cam, img_shape)
        # B*cam, nlvl*bH, bW, 2
        bev_grid_pos = bev_grid_pos.unflatten(-2, (nlvl*bH, bW))

        # project feat from 2D to bev plane
        projected_feature = F.grid_sample(
            cam_feat, bev_grid_pos).view(B, -1, C, nlvl, bH, bW)  # B,cam,C,nlvl,bH,bW

        # B,cam,nlvl,bH,bW
        bev_feat_mask = bev_cam_mask.unflatten(-1, (nlvl, bH, bW))

        # eliminate the ncam
        # The bev feature is the sum of the 6 cameras
        bev_feat_mask = bev_feat_mask.unsqueeze(2)
        projected_feature = (projected_feature*bev_feat_mask).sum(1)
        num_feat = bev_feat_mask.sum(1)

        projected_feature = projected_feature / \
            num_feat.masked_fill(num_feat == 0, 1)

        # concatenate a position information
        # projected_feature: B, bH, bW, nlvl, C+3
        bev_grid = bev_grid.view(B, nlvl, bH, bW,
                                 3).permute(0, 4, 1, 2, 3)
        projected_feature = torch.cat(
            (projected_feature, bev_grid), dim=1)

        bev_feat, bev_feat_mask = projected_feature, bev_feat_mask.sum(1) > 0

        # multi level into a same
        bev_feat = bev_feat.flatten(1, 2)
        bev_feat = self.outconvs(bev_feat)

        return bev_feat
