# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (FEEDFORWARD_NETWORK, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32
from mmcv.runner.base_module import BaseModule
from mmcv.utils import deprecated_api_warning, to_2tuple
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_


@FEEDFORWARD_NETWORK.register_module()
class EfficientFFN(BaseModule):

    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',
            'add_residual': 'add_identity'
        },
        cls_name='EfficientFFN')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 split=4,
                 use_checkpoint=False,
                 **kwargs):
        super(EfficientFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(ffn_drop)
        in_channels = embed_dims
        self.use_checkpoint = use_checkpoint
        self.split = split
        for i in range(split):
            fc1 = nn.Linear(in_channels, feedforward_channels //
                            self.split, bias=True)
            setattr(self, f'fc1_{i}', fc1)

        for i in range(split):
            fc2 = nn.Linear(feedforward_channels // self.split,
                            embed_dims, bias=False)
            setattr(self, f'fc2_{i}', fc2)
        self.fc2_bias = nn.Parameter(torch.zeros(
            (embed_dims)), requires_grad=True)
        # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.fc2_0.weight)
        # bound = 1 / math.sqrt(fan_in)
        # torch.nn.init.uniform_(self.fc2_bias, -bound, bound)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):

        def _inner_forward(x, i):
            fc1 = getattr(self, f'fc1_{i}')
            x = fc1(x)
            x = self.activate(x)
            x = self.drop(x)
            fc2 = getattr(self, f'fc2_{i}')
            x = fc2(x)
            x = self.drop(x)
            return x

        out = 0
        for i in range(self.split):
            if self.use_checkpoint and x.requires_grad:
                out = out + checkpoint.checkpoint(_inner_forward, x, i)
            else:
                out = out + _inner_forward(x, i)

        out = out + self.fc2_bias

        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
