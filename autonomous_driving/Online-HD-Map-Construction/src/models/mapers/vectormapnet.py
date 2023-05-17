import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.resnet import resnet18, resnet50

from mmdet3d.models.builder import (build_backbone, build_head,
                                    build_neck)

from .base_mapper import BaseMapper, MAPPERS


@MAPPERS.register_module()
class VectorMapNet(BaseMapper):

    def __init__(self,
                 backbone_cfg=dict(),
                 head_cfg=dict(
                     vert_net_cfg=dict(),
                     face_net_cfg=dict(),
                 ),
                 neck_input_channels=128,
                 neck_cfg=None,
                 with_auxiliary_head=False,
                 only_det=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 model_name=None, **kwargs):
        super(VectorMapNet, self).__init__()

        
        #Attribute
        self.model_name = model_name
        self.last_epoch = None
        self.only_det = only_det
  
        self.backbone = build_backbone(backbone_cfg)

        if neck_cfg is not None:
            self.neck_neck = build_backbone(neck_cfg.backbone)
            self.neck_neck.conv1 = nn.Conv2d(
                neck_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.neck_project = build_neck(neck_cfg.neck)
            self.neck = self.multiscale_neck
        else:
            trunk = resnet18(pretrained=False, zero_init_residual=True)
            self.neck = nn.Sequential(
                nn.Conv2d(neck_input_channels, 64, kernel_size=(7, 7), stride=(
                    2, 2), padding=(3, 3), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                            dilation=1, ceil_mode=False),
                trunk.layer1,
                nn.Conv2d(64, 128, kernel_size=1, bias=False),
            )
        
        # BEV 
        if hasattr(self.backbone,'bev_w'):
            self.bev_w = self.backbone.bev_w
            self.bev_h = self.backbone.bev_h


        self.head = build_head(head_cfg)

    def multiscale_neck(self, bev_embedding):

        multi_feat = self.neck_neck(bev_embedding)
        multi_feat = self.neck_project(multi_feat)

        return multi_feat

    def forward_train(self, img, polys, points=None, img_metas=None, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        #  prepare labels and images
        batch, img, img_metas, valid_idx, points = self.batch_data(
            polys, img, img_metas, img.device, points)
        
        # corner cases use hard code to prevent code fail
        if self.last_epoch is None:
            self.last_epoch = [batch, img, img_metas, valid_idx, points]

        if len(valid_idx)==0:
            batch, img, img_metas, valid_idx, points = self.last_epoch
        else:
            del self.last_epoch
            self.last_epoch = [batch, img, img_metas, valid_idx, points]

        # Backbone
        _bev_feats = self.backbone(img, img_metas=img_metas, points=points)
        img_shape = \
            [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]

        # Neck
        bev_feats = self.neck(_bev_feats)
        
        preds_dict, losses_dict = \
            self.head(batch, 
                      context={
                        'bev_embeddings': bev_feats, 
                        'batch_input_shape': _bev_feats.shape[2:], 
                        'img_shape': img_shape,
                        'raw_bev_embeddings': _bev_feats},
                        only_det=self.only_det)

        # format outputs
        loss = 0
        for name, var in losses_dict.items():
            loss = loss + var

        # update the log
        log_vars = {k: v.item() for k, v in losses_dict.items()}
        log_vars.update({'total': loss.item()})

        num_sample = img.size(0)

        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, polys=None, points=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        #  prepare labels and images
        token = []
        for img_meta in img_metas:
            token.append(img_meta['token'])

        _bev_feats = self.backbone(img, img_metas, points=points)
        img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]
        # Neck
        bev_feats = self.neck(_bev_feats)

        context = {'bev_embeddings': bev_feats,
                   'batch_input_shape': _bev_feats.shape[2:], 
                   'img_shape': img_shape, # XXX
                   'raw_bev_embeddings': _bev_feats}

        preds_dict = self.head(batch={},
                               context=context,
                               condition_on_det=True, 
                               gt_condition=False,
                               only_det=self.only_det)

        # Hard Code
        if preds_dict is None:
            return [None]

        results_list = self.head.post_process(preds_dict, token, only_det=self.only_det)

        return results_list

    def batch_data(self, polys, imgs, img_metas, device, points=None):
        # filter none vector's case
        valid_idx = [i for i in range(len(polys)) if len(polys[i])]
        imgs = imgs[valid_idx]
        img_metas = [img_metas[i] for i in valid_idx]
        
        polys = [polys[i] for i in valid_idx]

        if points is not None:
            points = [points[i] for i in valid_idx]
            points = self.batch_points(points)

        if len(valid_idx) == 0:
            return None, None, None, valid_idx, None

        batch = {}
        batch['det'] = format_det(polys,device)
        batch['gen'] = format_gen(polys,device)

        return batch, imgs, img_metas, valid_idx, points

    def batch_points(self, points):
    
        pad_points = pad_sequence(points, batch_first=True)

        points_mask = torch.zeros_like(pad_points[:,:,0]).bool()
        for i in range(len(points)):
            valid_num = points[i].shape[0]
            points_mask[i][:valid_num] = True

        return (pad_points, points_mask)


def format_det(polys, device):
    
    batch = {
        'class_label':[],
        'batch_idx':[],
        'bbox': [],
    }

    for batch_idx, poly in enumerate(polys):

        keypoint_label = torch.from_numpy(poly['det_label']).to(device)
        keypoint = torch.from_numpy(poly['keypoint']).to(device)
        
        batch['class_label'].append(keypoint_label)
        batch['bbox'].append(keypoint)
    
    return batch
        
 
def format_gen(polys,device):

    line_cls = []
    polylines, polyline_masks, polyline_weights = [], [], []
    bbox, line_cls, line_bs_idx = [], [], []
    
    for batch_idx, poly in enumerate(polys):
    
        # convert to cuda tensor
        for k in poly.keys():
            if isinstance(poly[k],np.ndarray):
                poly[k] = torch.from_numpy(poly[k]).to(device)
            else:
                poly[k] = [torch.from_numpy(v).to(device) for v in poly[k]]
        
        line_cls += poly['gen_label']
        line_bs_idx += [batch_idx]*len(poly['gen_label'])

        # condition
        bbox += poly['qkeypoint']

        # out
        polylines += poly['polylines']
        polyline_masks += poly['polyline_masks']
        polyline_weights += poly['polyline_weights']

    batch = {}
    batch['lines_bs_idx'] = torch.tensor(
        line_bs_idx, dtype=torch.long, device=device)
    batch['lines_cls'] = torch.tensor(
        line_cls, dtype=torch.long, device=device)
    batch['bbox_flat'] = torch.stack(bbox, 0)

    # padding
    batch['polylines'] = pad_sequence(polylines, batch_first=True)
    batch['polyline_masks'] = pad_sequence(polyline_masks, batch_first=True)
    batch['polyline_weights'] = pad_sequence(polyline_weights, batch_first=True)
    
    return batch