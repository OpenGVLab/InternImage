# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS

from .dino import DINO


@DETECTORS.register_module()
class CBDINO(DINO):

    def __init__(self, rule=None, **kwargs):
        super(CBDINO, self).__init__(**kwargs)
        for k, v in self.named_parameters():
            if rule == 'freeze_backbone_expect_level_4':
                if 'backbone' in k and 'backbone.cb_modules.1.levels.3' not in k:
                    v.requires_grad = False
            if rule == 'freeze_backbone_expect_0_level_1_2_3':
                if 'backbone' in k and 'backbone.cb_modules.0.levels.0' not in k \
                        and 'backbone.cb_modules.1.levels.0' not in k \
                        and 'backbone.cb_modules.2.levels.0' not in k:
                    v.requires_grad = False
            if rule == 'freeze_backbone_expect_level_3_4':
                if 'backbone' in k and 'backbone.cb_modules.1.levels.2' not in k \
                        and 'backbone.cb_modules.1.levels.3' not in k:
                    v.requires_grad = False
            if rule == 'freeze_cb_first_backbone':
                if 'backbone.cb_modules.0' in k:
                    v.requires_grad = False
            if rule == 'freeze_cb_first_backbone_expect_level_4':
                if 'backbone.cb_modules.0' in k and 'levels.3' not in k:
                    v.requires_grad = False
            if rule == 'freeze_backbone':
                if 'backbone' in k:
                    v.requires_grad = False
            if rule == 'freeze_backbone_encoder':
                if 'backbone' in k or 'encoder' in k:
                    v.requires_grad = False
            if rule == 'freeze_backbone_neck':
                if 'backbone' in k or 'neck' in k:
                    v.requires_grad = False
            if rule == 'freeze_stage_1_2':
                if 'patch_embed' in k:
                    v.requires_grad = False
                if 'levels.0.' in k or 'levels.1.' in k:
                    v.requires_grad = False
            if rule == 'freeze_stage_1':
                if 'patch_embed' in k:
                    v.requires_grad = False
                if 'levels.0.' in k:
                    v.requires_grad = False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      loss_weights=None,
                      **kwargs):

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        xs = self.extract_feat(img)
        # x0: x01, x02, x03, x04, x05
        # x1: x11, x12, x13, x14, x15

        if not isinstance(xs[0], (list, tuple)):
            xs = [xs]
            loss_weights = None
        elif loss_weights is None:
            loss_weights = [0.5] + [1]*(len(xs)-1)
            # [0.5, 1]

        losses = dict()
        new_x = [torch.cat((xs[0][i], xs[1][i])) for i in range(len(xs[0]))]
        img_metas = img_metas + img_metas
        gt_bboxes = gt_bboxes + gt_bboxes
        gt_labels = gt_labels + gt_labels
        gt_bboxes_ignore = gt_bboxes_ignore + \
            gt_bboxes_ignore if gt_bboxes_ignore is not None else None
        losses = self.bbox_head.forward_train(
            new_x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test_bboxes(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
