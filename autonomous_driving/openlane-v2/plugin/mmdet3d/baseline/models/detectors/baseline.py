# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# baseline.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

from mmdet3d.models import DETECTORS, build_neck, build_head
from mmdet3d.models.detectors import MVXTwoStageDetector


@DETECTORS.register_module()
class Baseline(MVXTwoStageDetector):

    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 lc_head=None,
                 te_head=None,
                 lclc_head=None,
                 lcte_head=None,
                 **kwargs):

        super().__init__(img_backbone=img_backbone, img_neck=img_neck, **kwargs)

        self.img_view_transformer = build_neck(img_view_transformer)
        self.lc_head = build_head(lc_head)
        self.te_head = build_head(te_head)
        self.lclc_head = build_head(lclc_head)
        self.lcte_head = build_head(lcte_head)

    def simple_forward(self, img, img_metas):

        # extract image features

        B, N, C, imH, imW = img.shape
        img = img.view(B * N, C, imH, imW)
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]

        # view transformation 

        bev_feat = self.img_view_transformer(
            x,
            torch.cat([
                torch.cat([torch.tensor(l, device=x.device, dtype=torch.float32).unsqueeze(0) for l in img_metas[b]['lidar2img']], dim=0).unsqueeze(0)
            for b in range(B)], dim=0),
            (img_metas[0]['img_shape'][0][0], img_metas[0]['img_shape'][0][1]),
        )
        _, output_dim, ouput_H, output_W = x.shape
        pv_feat = x.view(B, N, output_dim, ouput_H, output_W)[:, 0, ...]

        # lc

        lc_img_metas = [{
            'batch_input_shape': (bev_feat.shape[-2], bev_feat.shape[-1]),
            'img_shape': (bev_feat.shape[-2], bev_feat.shape[-1], None),
            'scale_factor': None, # dummy
        } for _ in range(B)]
        all_lc_cls_scores_list, all_lc_preds_list, lc_outs_dec_list = self.lc_head(
            [bev_feat], 
            lc_img_metas,
        )

        # te

        te_img_metas = [{
            'batch_input_shape': (img_metas[b]['pad_shape'][0][0], img_metas[b]['pad_shape'][0][1]),
            'img_shape': (img_metas[b]['img_shape'][0][0], img_metas[b]['img_shape'][0][1], None),
            'scale_factor': img_metas[b]['scale_factor'],
        } for b in range(B)]
        all_te_cls_scores_list, all_te_preds_list, te_outs_dec_list = self.te_head(
            [pv_feat], 
            te_img_metas,
        )

        # topology_lclc

        all_lclc_preds_list = self.lclc_head(
            lc_outs_dec_list,
            lc_outs_dec_list,
        )

        # topology_lcte

        all_lcte_preds_list = self.lcte_head(
            lc_outs_dec_list,
            te_outs_dec_list,
        )

        return {
            'all_lc_cls_scores_list': all_lc_cls_scores_list,
            'all_lc_preds_list': all_lc_preds_list,
            'lc_img_metas': lc_img_metas,
            'all_te_cls_scores_list': all_te_cls_scores_list,
            'all_te_preds_list': all_te_preds_list,
            'te_img_metas': te_img_metas,
            'all_lclc_preds_list': all_lclc_preds_list,
            'all_lcte_preds_list': all_lcte_preds_list,
        }

    def forward_train(self,
                      img,
                      img_metas,
                      gt_lc=None,
                      gt_lc_labels=None,
                      gt_te=None,
                      gt_te_labels=None,
                      gt_topology_lclc=None,
                      gt_topology_lcte=None,
                      **kwargs):

        outs = self.simple_forward(img, img_metas)

        losses = dict()

        # lc

        lc_loss_dict, lc_assign_results = self.lc_head.loss(
            outs['all_lc_cls_scores_list'],
            outs['all_lc_preds_list'],
            gt_lc,
            gt_lc_labels,
            outs['lc_img_metas'],
        )
        losses.update({
            f'lc_{key}': val for key, val in lc_loss_dict.items()
        })

        # te
        
        te_loss_dict, te_assign_results = self.te_head.loss(
            outs['all_te_cls_scores_list'],
            outs['all_te_preds_list'],
            gt_te,
            gt_te_labels,
            outs['te_img_metas'],
        )
        losses.update({
            f'te_{key}': val for key, val in te_loss_dict.items()
        })

        # topology_lclc

        topology_lclc_loss_dict = self.lclc_head.loss(
            outs['all_lclc_preds_list'],
            lc_assign_results,
            lc_assign_results,
            gt_topology_lclc,
        )
        losses.update({
            f'topology_lclc_{key}': val for key, val in topology_lclc_loss_dict.items()
        })

        # topology_lcte

        topology_lcte_loss_dict = self.lcte_head.loss(
            outs['all_lcte_preds_list'],
            lc_assign_results,
            te_assign_results,
            gt_topology_lcte
        )
        losses.update({
            f'topology_lcte_{key}': val for key, val in topology_lcte_loss_dict.items()
        })

        return losses
    
    def forward_test(self, img, img_metas, **kwargs):

        outs = self.simple_forward(img, img_metas)

        pred_lc = self.lc_head.get_bboxes(
            outs['all_lc_cls_scores_list'], 
            outs['all_lc_preds_list'], 
            outs['lc_img_metas'],
        )
        pred_te = self.te_head.get_bboxes(
            outs['all_te_cls_scores_list'], 
            outs['all_te_preds_list'], 
            outs['te_img_metas'], 
            rescale=True,
        )

        pred_topology_lclc = self.lclc_head.get_topology(outs['all_lclc_preds_list'])
        pred_topology_lcte = self.lcte_head.get_topology(outs['all_lcte_preds_list'])

        assert len(pred_lc) == len(pred_te) == 1, \
            'evaluation implemented for bs=1'
        return [{
            'pred_lc': pred_lc[0],
            'pred_te': pred_te[0],
            'pred_topology_lclc': pred_topology_lclc[0],
            'pred_topology_lcte': pred_topology_lcte[0],
        }]
