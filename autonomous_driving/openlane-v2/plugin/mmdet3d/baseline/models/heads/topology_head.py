# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# topology_head.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet.models import HEADS, build_loss


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class TopologyHead(BaseModule):

    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 loss_cls):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = MLP(self.in_channels, hidden_channels, out_channels, num_layers)
        self.loss_cls = build_loss(loss_cls)

    def forward(self, all_a_preds_list, all_b_preds_list):

        # NOTE defaultly only the outputs from the last feature scale is used.
        all_a_preds_list = all_a_preds_list[-1]
        all_b_preds_list = all_b_preds_list[-1]

        num_out = all_a_preds_list.shape[0]
        assert num_out == all_a_preds_list.shape[0] == all_b_preds_list.shape[0]

        num_row, num_column = all_a_preds_list.shape[-2], all_b_preds_list.shape[-2]
        assert self.in_channels == all_a_preds_list.shape[-1] + all_b_preds_list.shape[-1], \
            f'self.in_channels = {self.in_channels} != all_a_preds_list.shape[-1] {all_a_preds_list.shape[-1]} + all_b_preds_list.shape[-1] {all_b_preds_list.shape[-1]}'

        outs = []
        for o in range(num_out):
            adj = torch.cat([
                all_a_preds_list[o].unsqueeze(2).repeat(1, 1, num_column, 1),
                all_b_preds_list[o].unsqueeze(1).repeat(1, num_row, 1, 1),
            ], dim=-1)

            outs.append(self.mlp(adj).sigmoid())

        return outs

    def loss(self, pred_adj_list, row_assign_results, column_assign_results, gt_adj):

        # NOTE defaultly only the outputs from the last decoder layer is used.
        pred_adj = pred_adj_list[-1]
        row_assign_result = row_assign_results[-1]
        column_assign_result = column_assign_results[-1]

        targets = []
        for b in range(pred_adj.shape[0]):
            target = pred_adj.new_zeros(pred_adj[b].shape[:-1])
            rs = row_assign_result['pos_inds'][b].unsqueeze(-1).repeat(1, column_assign_result['pos_inds'][b].shape[0])
            cs = column_assign_result['pos_inds'][b].unsqueeze(0).repeat(row_assign_result['pos_inds'][b].shape[0], 1)
            target[rs, cs] = gt_adj[b][row_assign_result['pos_assigned_gt_inds'][b]][:, column_assign_result['pos_assigned_gt_inds'][b]].float()
            targets.append(target)

        targets = 1 - torch.stack(targets, dim=0) # 0 as positive

        loss_dict = dict()
        pred_adj = pred_adj.reshape(-1, self.out_channels)
        targets = targets.long().reshape(-1)
        loss_dict['loss_cls'] = self.loss_cls(pred_adj, targets)
        return loss_dict

    def get_topology(self, pred_adj_list):
        # NOTE defaultly only the outputs from the last decoder layer is used.
        pred_adj = pred_adj_list[-1].squeeze(-1)
        return pred_adj.cpu().numpy()
