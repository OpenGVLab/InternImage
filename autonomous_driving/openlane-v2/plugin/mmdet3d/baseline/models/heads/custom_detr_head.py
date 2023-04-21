# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# custom_detr_head.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
import torch.nn.functional as F

from mmcv.cnn import Linear
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply, reduce_mean
from mmdet.models import HEADS, DETRHead


@HEADS.register_module()
class CustomDETRHead(DETRHead):
    
    def __init__(self, 
                 num_classes, 
                 in_channels,
                 num_query,
                 object_type,
                 num_reg_dim=4,
                 num_layers=1,
                 feedforward_channels=512,
                 embed_dims=64,
                 num_heads=4,
                 dropout=0.1,
                 ffn_dropout=0.1,
                 **kwargs):

        self.object_type = object_type 
        if self.object_type == 'lane':
            self.num_reg_dim = num_reg_dim
            assert self.num_reg_dim % 3 == 0
            self.bev_range = kwargs['bev_range']
        elif self.object_type == 'bbox':
            self.num_reg_dim = 4
            assert self.num_reg_dim == num_reg_dim == 4
        else:
            raise NotImplementedError

        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=num_layers,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            dropout=dropout)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=feedforward_channels,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=feedforward_channels,
                    ffn_dropout=ffn_dropout,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=num_layers,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        dropout=dropout),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=feedforward_channels,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=feedforward_channels,
                    ffn_dropout=ffn_dropout,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            ))
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=embed_dims//2, normalize=True)
        super().__init__(
            num_classes=num_classes, 
            in_channels=in_channels, 
            num_query=num_query, 
            transformer=transformer, 
            positional_encoding=positional_encoding,
            **kwargs,
        )

    def _init_layers(self):
        super()._init_layers()
        self.fc_reg = Linear(self.embed_dims, self.num_reg_dim)

    def forward_single(self, x, img_metas):

        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        if self.object_type == 'lane':
            masks = x.new_zeros((x.shape[0], x.shape[2], x.shape[3]))
        else:
            batch_size = x.size(0)
            input_img_h, input_img_w = img_metas[0]['batch_input_shape']
            masks = x.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w, _ = img_metas[img_id]['img_shape']
                masks[img_id, :img_h, :img_w] = 0
                
        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds, outs_dec

    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):

        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, assign_result = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        if self.object_type != 'lane':
            loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            if self.object_type != 'lane':
                loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict, assign_result

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        if self.object_type == 'lane':
            bbox_preds = bbox_preds.reshape(-1, self.num_reg_dim)
            loss_iou = None
        else:
            # construct factors used for rescale bboxes
            factors = []
            for img_meta, bbox_pred in zip(img_metas, bbox_preds):
                img_h, img_w, _ = img_meta['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0).repeat(
                                                bbox_pred.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors, 0)

            # DETR regress the relative position of boxes (cxcywh) in the image,
            # thus the learning target is normalized by the image size. So here
            # we need to re-scale them for calculating IoU loss
            bbox_preds = bbox_preds.reshape(-1, self.num_reg_dim)
            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            # regression IoU loss, defaultly GIoU loss
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, assign_result

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, assign_result)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        if self.object_type == 'lane':
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            pos_gt_bboxes_normalized = torch.zeros_like(pos_gt_bboxes)
            for p in range(self.num_reg_dim // 3):
                pos_gt_bboxes_normalized[..., 3*p] = (pos_gt_bboxes[..., 3*p] - self.bev_range[0]) / (self.bev_range[3] - self.bev_range[0])
                pos_gt_bboxes_normalized[..., 3*p+1] = (pos_gt_bboxes[..., 3*p+1] - self.bev_range[1]) / (self.bev_range[4] - self.bev_range[1])
                pos_gt_bboxes_normalized[..., 3*p+2] = (pos_gt_bboxes[..., 3*p+2] - self.bev_range[2]) / (self.bev_range[5] - self.bev_range[2])
            pos_gt_bboxes_targets = pos_gt_bboxes_normalized
        else:
            img_h, img_w, _ = img_meta['img_shape']

            # DETR regress the relative position of boxes (cxcywh) in the image.
            # Thus the learning target should be normalized by the image size, also
            # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
            pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
            
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_assigned_gt_inds)

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(cls_score) == len(bbox_pred)

        if self.object_type == 'lane':
            cls_score = cls_score.sigmoid()

            det_bboxes = bbox_pred
            for p in range(self.num_reg_dim // 3):
                det_bboxes[..., 3*p] = det_bboxes[..., 3*p] * (self.bev_range[3] - self.bev_range[0]) + self.bev_range[0]
                det_bboxes[..., 3*p+1] = det_bboxes[..., 3*p+1] * (self.bev_range[4] - self.bev_range[1]) + self.bev_range[1]
                det_bboxes[..., 3*p+2] = det_bboxes[..., 3*p+2] * (self.bev_range[5] - self.bev_range[2]) + self.bev_range[2]
                det_bboxes[..., 3*p].clamp_(min=self.bev_range[0], max=self.bev_range[3])
                det_bboxes[..., 3*p+1].clamp_(min=self.bev_range[1], max=self.bev_range[4])
                det_bboxes[..., 3*p+2].clamp_(min=self.bev_range[2], max=self.bev_range[5])
        else:
            # exclude background
            if self.loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
            else:
                cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
            cls_score, det_labels = cls_score.max(-1)

            det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
            det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
            if rescale:
                det_bboxes /= det_bboxes.new_tensor(scale_factor)
            det_bboxes = torch.cat((det_bboxes, det_labels.unsqueeze(1)), -1)

        return det_bboxes.cpu().numpy(), cls_score.cpu().numpy()

    def onnx_export(self, **kwargs):
        raise NotImplementedError(f'TODO: replace 4 with self.num_reg_dim : {self.num_reg_dim}')
