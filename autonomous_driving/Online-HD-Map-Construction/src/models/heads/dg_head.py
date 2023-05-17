import copy
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.models import HEADS, build_head, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from .base_map_head import BaseMapHead

import numpy as np
from ..augmentation.sythesis_det import NoiseSythesis

@HEADS.register_module(force=True)
class DGHead(BaseMapHead):

    def __init__(self,
                 det_net_cfg=dict(),
                 gen_net_cfg=dict(),
                 loss_vert=dict(),
                 loss_face=dict(),
                 max_num_vertices=90,
                 top_p_gen_model=0.9,
                 sync_cls_avg_factor=True,
                 augmentation=False,
                 augmentation_kwargs=None,
                 joint_training=False,
                 **kwargs):
        super().__init__()

        # Heads
        self.det_net = build_head(det_net_cfg)
        self.gen_net = build_head(gen_net_cfg)

        self.coord_dim = self.gen_net.coord_dim

        # Loss params
        self.bg_cls_weight = 1.0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.max_num_vertices = max_num_vertices
        self.top_p_gen_model = top_p_gen_model

        self.fp16_enabled = False

        self.augmentation = None
        if augmentation:
            augmentation_kwargs.update({'canvas_size':gen_net_cfg.canvas_size})
            self.augmentation = NoiseSythesis(**augmentation_kwargs)
        
        self.joint_training = joint_training

    def forward(self, batch, img_metas=None, **kwargs):
        '''
            Args:
            Returns:
                outs (Dict):   
        '''

        if self.training:
            return self.forward_train(batch, **kwargs)
        else:
            return self.inference(batch, **kwargs)

    def forward_train(self, batch: dict, context: dict, only_det=False, **kwargs):
        ''' we use teacher force strategy'''

        bbox_dict = self.det_net(context=context)
        outs = dict(
                bbox=bbox_dict,
            )

        losses_dict, det_match_idxs, det_match_gt_idxs = \
            self.loss_det(batch, outs)

        if only_det: return outs, losses_dict

        if self.augmentation is not None:
            polylines, bbox_flat =\
                self.augmentation(batch['gen'],simple_aug=True)

            if bbox_flat is None:
                bbox_flat = batch['gen']['bbox_flat']
            
            gen_input = dict(
                lines_bs_idx=batch['gen']['lines_bs_idx'],
                lines_cls=batch['gen']['lines_cls'],
                bbox_flat=bbox_flat,
                polylines=polylines,
                polyline_masks=batch['gen']['polyline_masks']
            )
        else:
            gen_input = batch['gen']

        if self.joint_training:

            # for down stream polyline
            if 'lines' in bbox_dict[-1]:
                # for fix anchor
                pred_bbox = bbox_dict[-1]['lines'].detach()
            elif 'bboxs' in bbox_dict[-1]:
                # for rpv
                pred_bbox = bbox_dict[-1]['bboxs'].detach()
            else:
                raise NotImplementedError
        
            #  changed to original gt order. 
            det_match_idx = det_match_idxs[-1]
            det_match_gt_idx = det_match_gt_idxs[-1]

            _bboxs = []
            for i, (match_idx, bbox) in enumerate(zip(det_match_idx,pred_bbox)):
                    _bboxs.append(bbox[match_idx])
                    _bboxs[-1] = _bboxs[-1][torch.argsort(det_match_gt_idx[i])]

            _bboxs = torch.cat(_bboxs, dim=0)
            
            # quantize the data
            _bboxs = \
                torch.round(_bboxs).type(torch.int32)

            # gen_input['bbox_flat'] = _bboxs
            remain_idx = torch.randperm(_bboxs.shape[0])[:int(_bboxs.shape[0]*0.2)]
            # for data efficient
            for k in gen_input.keys():
                if k == 'bbox_flat':
                    gen_input[k] = torch.cat((_bboxs,gen_input[k][remain_idx]),dim=0)
                else:
                    gen_input[k] = torch.cat((gen_input[k],gen_input[k][remain_idx]),dim=0)
    
        if isinstance(context['bev_embeddings'],tuple):
            context['bev_embeddings'] = context['bev_embeddings'][0]

        poly_dict = self.gen_net(gen_input, context=context)

        outs.update(dict(
            polylines=poly_dict,
        ))

        if self.joint_training:
            for k in batch['gen'].keys():
                batch['gen'][k] = \
                    torch.cat((batch['gen'][k],batch['gen'][k][remain_idx]),dim=0)

        gen_losses_dict = \
            self.loss_gen(batch, outs)

        losses_dict.update(gen_losses_dict) 

        return outs, losses_dict

    def loss_det(self, gt: dict, pred: dict):
        
        loss_dict = {}

        # det
        det_loss_dict, det_match_idx, det_match_gt_idx = \
            self.det_net.loss(gt['det'], pred['bbox'])

        for k, v in det_loss_dict.items():
            loss_dict['det_'+k] = v
        
        return loss_dict, det_match_idx, det_match_gt_idx

    def loss_gen(self, gt: dict, pred: dict):

        loss_dict = {}

        # gen
        gen_loss_dict = self.gen_net.loss(gt['gen'], pred['polylines'])

        for k, v in gen_loss_dict.items():
            loss_dict['gen_'+k] = v

        return loss_dict
    
    def loss(self, gt: dict, pred: dict):
        
        pass

    @torch.no_grad()
    def inference(self, batch: dict={}, context: dict={}, gt_condition=False, **kwargs):
        '''
            num_samples_batch: number of sample per batch (batch size)
        '''
        outs = {}
        bbox_dict = self.det_net(context=context)
        bbox_dict = self.det_net.post_process(bbox_dict)
        
        outs.update(bbox_dict)
        
        if len(outs['lines_bs_idx']) == 0:
            return None
        
        if isinstance(context['bev_embeddings'],tuple):
            context['bev_embeddings'] = context['bev_embeddings'][0]

        poly_dict = self.gen_net(outs,
                                 context=context,
                                #  max_sample_length=self.max_num_vertices,
                                 max_sample_length=64,
                                 top_p=self.top_p_gen_model,
                                 gt_condition=gt_condition)
        outs.update(poly_dict)

        return outs

    def post_process(self, preds: dict, tokens, gts:dict=None, **kwargs):
        '''
            Args:
                XXX
            Outs:
               XXX
        '''
        range_size = self.gen_net.canvas_size.cpu().numpy()
        coord_dim = self.gen_net.coord_dim
        
        gen_net_name = self.gen_net.name if hasattr(self.gen_net,'name') else 'gen'

        ret_list = []
        for batch_idx in range(len(tokens)):

            ret_dict_single = {}

            # bbox
            det_gt = None
            if gts is not None:
                det_gt, rec_groundtruth = pack_groundtruth(
                    batch_idx,gts,tokens,range_size,gen_net_name,coord_dim=coord_dim)
                
            bbox_res = {
                # 'bboxes': preds['bbox'][batch_idx].detach().cpu().numpy(),
                # 'det_gt': det_gt,
                'token': tokens[batch_idx],
                'scores': preds['scores'][batch_idx].detach().cpu().numpy(),
                'labels': preds['labels'][batch_idx].detach().cpu().numpy(),
            }
            ret_dict_single.update(bbox_res)


            # for gen results.
            batch2seq = np.nonzero(
                preds['lines_bs_idx'].cpu().numpy() == batch_idx)[0]

            ret_dict_single.update({
                'nline': len(batch2seq),
                'vectors': []
            })

            for i in batch2seq:

                pre = preds['polylines'][i].detach().cpu().numpy()
                pre_msk = preds['polyline_masks'][i].detach().cpu().numpy()
                valid_idx = np.nonzero(pre_msk)[0][:-1]

                # From [200,1] to [199,0] to (1,0)
                line = (pre[valid_idx].reshape(-1, coord_dim) - 1) / (range_size-1)

                ret_dict_single['vectors'].append(line)
        
            # if gts is not None:
            #     ret_dict_single['groundTruth'] = rec_groundtruth

            ret_list.append(ret_dict_single)

        return ret_list

def pack_groundtruth(batch_idx,gts,tokens,range_size,gen_net_name='gen',coord_dim=2):

    if 'keypoints' in gts['det']:
        gt_bbox = \
            gts['det']['keypoints'][batch_idx].detach().cpu().numpy()
    else:
        gt_bbox = \
            gts['det']['bbox'][batch_idx].detach().cpu().numpy()
    det_gt = {
        'labels': gts['det']['class_label'][batch_idx].detach().cpu().numpy(),
        'bboxes': gt_bbox,
    }

    batch2seq = np.nonzero(
        gts['gen']['lines_bs_idx'].cpu().numpy() == batch_idx)[0]
    
    ret_groundtruth = {
        'token': tokens[batch_idx],
        'nline': len(batch2seq),
        'labels': gts['gen']['lines_cls'][batch2seq].detach().cpu().numpy(),
        'lines': [],
    }

    for i in batch2seq:
        gt_line =\
            gts['gen']['polylines'].detach().cpu().numpy()[i]
        gt_msk = gts['gen']['polyline_masks'].detach().cpu().numpy()[i]
        if gen_net_name == 'gen_gmm':
            valid_idx = np.nonzero(gt_msk)[0]
        else:
            valid_idx = np.nonzero(gt_msk)[0][:-1]
        
        # From [200,1] to [199,0] to (1,0)
        line = (gt_line[valid_idx].reshape(-1, coord_dim) - 1) / (range_size-1)
        ret_groundtruth['lines'].append(line)
    
    return det_gt, ret_groundtruth
