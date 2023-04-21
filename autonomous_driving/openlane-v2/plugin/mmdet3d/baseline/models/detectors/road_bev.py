# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Tianyu Li
# ---------------------------------------------
import time
import copy
import numpy as np
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import bbox2result
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.builder import build_neck
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class ROAD_BEVFormer(MVXTwoStageDetector):

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 bev_constructor=None,
                 bbox_head=None,
                 bbox_train_cfg=None,
                 lclc_head=None,
                 lcte_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(ROAD_BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        if bev_constructor is not None:
            self.bev_constructor = build_neck(bev_constructor)

        if bbox_head is not None:
            bbox_head.update(train_cfg=bbox_train_cfg)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        if lclc_head is not None:
            self.lclc_head = build_head(lclc_head)
        else:
            self.lclc_head = None

        if lcte_head is not None:
            self.lcte_head = build_head(lcte_head)
        else:
            self.lcte_head = None

        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_constructor(img_feats, img_metas, prev_bev)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_te=None,
                      gt_te_labels=None,
                      gt_lc=None,
                      gt_lc_labels=None,
                      gt_topology_lclc=None,
                      gt_topology_lcte=None,
                      gt_bboxes_ignore=None,
                      ):
        prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)

        losses = dict()
        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas)
        loss_inputs = [outs, gt_lc, gt_lc_labels]
        lane_losses, lane_assign_result = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]
        lane_feats = outs['history_states']

        if self.lclc_head is not None:
            lclc_losses = self.lclc_head.forward_train(lane_feats, lane_assign_result, lane_feats, lane_assign_result, gt_topology_lclc)
            for loss in lclc_losses:
                losses['lclc_head.' + loss] = lclc_losses[loss]

        if self.bbox_head is not None:
            front_view_img_feats = [lvl[:, 0] for lvl in img_feats]
            batch_input_shape = tuple(img[0, 0].size()[-2:])
            bbox_img_metas = []
            for img_meta in img_metas:
                bbox_img_metas.append(
                    dict(
                        batch_input_shape=batch_input_shape,
                        img_shape=img_meta['img_shape'][0],
                        scale_factor=img_meta['scale_factor'][0]))
                img_meta['batch_input_shape'] = batch_input_shape

            te_losses = {}
            bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)
            bbox_losses, te_assign_result = self.bbox_head.loss(bbox_outs, gt_te, gt_te_labels, bbox_img_metas, gt_bboxes_ignore)
            for loss in bbox_losses:
                te_losses['bbox_head.' + loss] = bbox_losses[loss]

            if self.lcte_head is not None:
                te_feats = bbox_outs['history_states']
                lcte_losses = self.lcte_head.forward_train(lane_feats, lane_assign_result, te_feats, te_assign_result, gt_topology_lcte)
                for loss in lcte_losses:
                    te_losses['lcte_head.' + loss] = lcte_losses[loss]

            num_gt_bboxes = sum([len(gt) for gt in gt_te_labels])
            if num_gt_bboxes == 0:
                for loss in te_losses:
                    te_losses[loss] *= 0

            losses.update(te_losses)

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=None, **kwargs)
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)

        bev_feats = self.bev_constructor(x, img_metas, prev_bev)
        outs = self.pts_bbox_head(x, bev_feats, img_metas)

        lane_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        if self.lclc_head is not None:
            lane_feats = outs['history_states']
            lclc_results = self.lclc_head.get_relationship(lane_feats, lane_feats)
            lclc_results = [result.detach().cpu().numpy() for result in lclc_results]
        else:
            lclc_results = [None for _ in range(batchsize)]

        if self.bbox_head is not None:
            front_view_img_feats = [lvl[:, 0] for lvl in x]
            batch_input_shape = tuple(img[0, 0].size()[-2:])
            bbox_img_metas = []
            for img_meta in img_metas:
                bbox_img_metas.append(
                    dict(
                        batch_input_shape=batch_input_shape,
                        img_shape=img_meta['img_shape'][0],
                        scale_factor=img_meta['scale_factor'][0]))
                img_meta['batch_input_shape'] = batch_input_shape
            bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)
            bbox_results = self.bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
        else:
            bbox_results = [None for _ in range(batchsize)]
        
        if self.bbox_head is not None and self.lcte_head is not None:
            te_feats = bbox_outs['history_states']
            lcte_results = self.lcte_head.get_relationship(lane_feats, te_feats)
            lcte_results = [result.detach().cpu().numpy() for result in lcte_results]
        else:
            lcte_results = [None for _ in range(batchsize)]

        return bev_feats, bbox_results, lane_results, lclc_results, lcte_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_results, lane_results, lclc_results, lcte_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        for result_dict, bbox, lane, lclc, lcte in zip(results_list, bbox_results, lane_results, lclc_results, lcte_results):
            result_dict['pred_te'] = bbox
            result_dict['pred_lc'] = lane
            result_dict['pred_topology_lclc'] = lclc
            result_dict['pred_topology_lcte'] = lcte
        return new_prev_bev, results_list
