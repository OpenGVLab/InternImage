import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmdet.models import HEADS
from mmcv.cnn import Conv2d
from mmcv.cnn import Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmcv.runner import force_fp32

from mmdet.core import (multi_apply, build_assigner, build_sampler,
                        reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import build_loss

from .base_map_head import BaseMapHead


@HEADS.register_module()
class DETRMapFixedNumHead(BaseMapHead):

    def __init__(self,
                 num_classes=3,
                 in_channels=128,
                 num_query=100,
                 max_lines=50,
                 score_thre=0.2,
                 num_reg_fcs=2,
                 num_points=100,
                 iterative=False,
                 patch_size=None,
                 sync_cls_avg_factor=True,
                 transformer: dict = None,
                 positional_encoding: dict = None,
                 loss_cls: dict = None,
                 loss_reg: dict = None,
                 train_cfg: dict = None,
                 init_cfg=None,
                 **kwargs):
        super().__init__()

        assigner = train_cfg['assigner']
        self.assigner = build_assigner(assigner)
        # DETR sampling=False, so use PseudoSampler
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.train_cfg = train_cfg
        self.max_lines = max_lines
        self.score_thre = score_thre

        self.num_query = num_query
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_points = num_points

        # branch
        # if loss_cls.use_sigmoid:
        if loss_cls['use_sigmoid']:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes+1

        self.iterative = iterative
        self.num_reg_fcs = num_reg_fcs

        if patch_size is not None:
            self.register_buffer('patch_size', torch.tensor(
                (patch_size[1], patch_size[0])),)

        self._build_transformer(transformer, positional_encoding)

        # loss params
        self.loss_cls = build_loss(loss_cls)
        self.bg_cls_weight = 0.1
        if self.loss_cls.use_sigmoid:
            self.bg_cls_weight = 0.0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.reg_loss = build_loss(loss_reg)

        # add reg, cls head for each decoder layer
        self._init_layers()
        self._init_branch()
        self.init_weights()

    def _init_layers(self):
        """Initialize some layer."""

        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        # query_pos_embed & query_embed
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims)

    def _build_transformer(self, transformer, positional_encoding):
        # transformer
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.num_points*2))
        reg_branch = nn.Sequential(*reg_branch)
        # add sigmoid or not

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers

        if self.iterative:
            fc_cls = _get_clones(fc_cls, num_pred)
            reg_branch = _get_clones(reg_branch, num_pred)

        self.pre_branches = nn.ModuleDict([
            ('cls', fc_cls),
            ('reg', reg_branch), ])

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.transformer.init_weights()

        # init prediction branch
        for k, v in self.pre_branches.items():
            for param in v.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        # focal loss init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            # for last layer
            if isinstance(self.pre_branches['cls'], nn.ModuleList):
                for m in self.pre_branches['cls']:
                    nn.init.constant_(m.bias, bias_init)
            else:
                m = self.pre_branches['cls']
                nn.init.constant_(m.bias, bias_init)

    def forward(self, bev_feature, img_metas=None):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
            img_metas
        Outs:
            preds_dict (Dict):
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lines_preds (Tensor):
                    [nb_dec, bs, num_query, num_points, 2].
        '''

        x = bev_feature[0]
        x = self.input_proj(x)  # only change feature size
        B, C, H, W = x.shape

        masks = x.new_zeros((B, H, W))
        pos_embed = self.positional_encoding(masks)
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks.type(torch.bool), self.query_embedding.weight,
                                       pos_embed)

        outputs = []

        for i, query_feat in enumerate(outs_dec):

            ocls = self.pre_branches['cls'](query_feat)
            oreg = self.pre_branches['reg'](query_feat)
            oreg = oreg.unflatten(dim=2, sizes=(self.num_points, 2))
            oreg[..., 0:2] = oreg[..., 0:2].sigmoid()  # normalized xyz

            outputs.append(
                dict(
                    lines=oreg,  # [bs, num_query, num_points, 2]
                    scores=ocls,  # [bs, num_query, num_class]
                )
            )

        return outputs

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_lines,
                           gt_labels,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                cls_score (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, num_points, 2].
                gt_lines (Tensor):
                    shape [num_gt, num_points, 2].
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
            Returns:
                tuple[Tensor]: a tuple containing the following for one image.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_pred_lines = lines_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
                                             gts=dict(lines=gt_lines,
                                                      labels=gt_labels, ),
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_lines.new_full((num_pred_lines, ),
                                   self.num_classes,
                                   dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines)

        # bbox targets
        lines_target = torch.zeros_like(lines_pred)
        lines_target[pos_inds] = sampling_result.pos_gt_bboxes

        lines_weights = torch.zeros_like(lines_pred)
        lines_weights[pos_inds] = 1.0

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds)

    @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                cls_scores_list (list[Tensor]): Box score logits from a single
                    decoder layer for each image with shape [num_query,
                    cls_out_channels].
                lines_preds_list (list[Tensor]): [num_query, num_points, 2].
                gt_lines_list (list[Tensor]): Ground truth lines for each image
                    with shape (num_gts, num_points, 2)
                gt_labels_list (list[Tensor]): Ground truth class indices for each
                    image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        (labels_list, label_weights_list,
         lines_targets_list, lines_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             preds['scores'], preds['lines'],
             gts['lines'], gts['labels'],
             gt_bboxes_ignore=gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list,
            label_weights=label_weights_list,
            lines_targets=lines_targets_list,
            lines_weights=lines_weights_list,
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list

    @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds: dict,
                    gts: dict,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """ 
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                cls_scores (Tensor): Box score logits from a single decoder layer
                    for all images. Shape [bs, num_query, cls_out_channels].
                lines_preds (Tensor):
                    shape [bs, num_query, num_points, 2].
                gt_lines_list (list[Tensor]): 
                    with shape (num_gts, num_points, 2)
                gt_labels_list (list[Tensor]): Ground truth class indices for each
                    image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list =\
            self.get_targets(preds, gts, gt_bboxes_ignore_list)

        # batched all data
        for k, v in new_gts.items():
            new_gts[k] = torch.cat(v, 0)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # classification loss
        cls_scores = preds['scores'].reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_scores, new_gts['labels'], new_gts['label_weights'], avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lines_preds = preds['lines'].reshape(-1, self.num_points, 2)
        if reduction == 'none':  # For performance analysis
            loss_reg = self.reg_loss(
                lines_preds, new_gts['lines_targets'], new_gts['lines_weights'], reduction_override=reduction, avg_factor=num_total_pos)
        else:
            loss_reg = self.reg_loss(
                lines_preds, new_gts['lines_targets'], new_gts['lines_weights'], avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return (loss_dict, pos_inds_list)

    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts: dict,
             preds_dicts: dict,
             gt_bboxes_ignore=None,
             reduction='mean'):
        """
            Loss Function.
            Args:
                gt_lines_list (list[Tensor]): Ground truth lines for each image
                    with shape (num_gts, num_points, 2)
                gt_labels_list (list[Tensor]): Ground truth class indices for each
                    image with shape (num_gts, ).
                preds_dicts:
                    all_cls_scores (Tensor): Classification score of all
                        decoder layers, has shape
                        [nb_dec, bs, num_query, cls_out_channels].
                    all_lines_preds (Tensor):
                        [nb_dec, bs, num_query, num_points, 2].
                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # Since there might have multi layer
        losses, pos_inds_lists, pos_gt_inds_lists = multi_apply(
            self.loss_single,
            preds_dicts,
            gts=gts,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v

        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists

    def post_process(self, preds_dict, tokens, gts):
        '''
        Args:
            preds_dict:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lines_preds (Tensor):
                    [nb_dec, bs, num_query, num_points, 2].
        Outs:
            ret_list (List[Dict]) with length as bs
                list of result dict for each sample in the batch
                Dict keys:
                'lines': numpy.array of shape [num_pred, num_points, 2]
                'scores': numpy.array of shape [num_pred, ]
                    after sigmoid
                'labels': numpy.array of shape [num_pred, ]
                    dtype=long
        '''

        preds = preds_dict[-1]

        batched_cls_scores = preds['scores']
        batched_lines_preds = preds['lines']
        batch_size = batched_cls_scores.size(0)

        ret_list = []
        for i in range(len(tokens)):

            cls_scores = batched_cls_scores[i]
            lines_preds = batched_lines_preds[i]
            max_num = self.max_lines

            if cls_scores.shape[-1] > self.num_classes:
                scores, labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
                final_scores, bbox_index = scores.topk(self.max_lines)
                final_lines = lines_preds[bbox_index]
                final_labels = labels[bbox_index]
            else:
                cls_scores = cls_scores.sigmoid()
                final_scores, indexes = cls_scores.view(-1).topk(self.max_lines)
                final_labels = indexes % self.num_classes
                bbox_index = indexes // self.num_classes
                final_lines = lines_preds[bbox_index]

            ret_dict_single = {
                'token': tokens[i],
                'lines': final_lines.detach().cpu().numpy() * 2 - 1,
                'scores': final_scores.detach().cpu().numpy(),
                'labels': final_labels.detach().cpu().numpy(),
                'nline': len(final_lines),
            }

            if gts is not None:
                lines_gt = gts['lines'][i].detach().cpu().numpy()
                labels_gt = gts['labels'][i].detach().cpu().numpy()
                ret_dict_single['groundTruth'] = {
                    'token': tokens[i],
                    'nline': lines_gt.shape[0],
                    'labels': labels_gt,
                    'lines': lines_gt * 2 - 1,
                }
                # if (labels_gt==1).any():
                #     import ipdb; ipdb.set_trace()

            ret_list.append(ret_dict_single)

        return ret_list
