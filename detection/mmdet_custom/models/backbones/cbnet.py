# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm

from .intern_image import InternImage


class LayerScale(nn.Module):
    def __init__(self, init_values=0., dim=1024):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        return self.gamma * x


class _InternImage(InternImage):

    def __init__(self, cb_idx, **kwargs):
        super(_InternImage, self).__init__(**kwargs)
        self.cb_idx = cb_idx
        self.num_features_list = [int(self.channels * 2 ** i) for i in range(self.num_levels)]
        if cb_idx == 1:
            self.gamma0 = nn.Parameter(torch.zeros((self.num_features_list[0])), requires_grad=True)
            self.gamma1 = nn.Parameter(torch.zeros((self.num_features_list[1])), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.zeros((self.num_features_list[2])), requires_grad=True)
            self.gamma3 = nn.Parameter(torch.zeros((self.num_features_list[3])), requires_grad=True)

    def del_layers(self, del_stages):
        self.del_stages = del_stages
        if self.del_stages >= 0:
            del self.patch_embed

    def forward(self, x, cb_feats=None, pre_tmps=None):
        outs, tmps = [], []
        if hasattr(self, 'patch_embed'):
            x = self.patch_embed(x)
            x = self.pos_drop(x)
            Wh, Ww = x.size(1), x.size(2)
            tmps.append((x, Wh, Ww))
        else:
            x, Wh, Ww = pre_tmps[0]

        for i, level in enumerate(self.levels):
            if cb_feats is not None:
                gamma = getattr(self, f'gamma{i}')
                x = x + gamma.half() * cb_feats[i]  # [B, H, W, C]
            x, x_ = level(x, return_wo_downsample=True)

            if i in self.out_indices:
                outs.append(x_.permute(0, 3, 1, 2).contiguous())

        return tuple(outs), tmps

    def train(self, mode=True):
        super(_InternImage, self).train(mode)


@BACKBONES.register_module()
class CBInternImage(BaseModule):
    def __init__(self, channels=96, out_indices=None, cb_zero_init=True, cb_del_stages=1, **kwargs):
        super(CBInternImage, self).__init__()
        self.cb_zero_init = cb_zero_init
        self.cb_del_stages = cb_del_stages
        self.out_indices = out_indices
        assert len(out_indices) == 2
        self.cb_modules = nn.ModuleList()
        for cb_idx in range(2):
            cb_module = _InternImage(channels=channels,
                                      out_indices=out_indices[cb_idx],
                                      cb_idx=cb_idx, **kwargs)
            if cb_idx > 0:
                cb_module.del_layers(cb_del_stages)
            self.cb_modules.append(cb_module)

        self.num_layers = self.cb_modules[0].num_layers

        cb_inplanes = [channels * 2 ** i for i in range(self.num_layers)]
        self.cb_linears = nn.ModuleList()

        for i in range(self.num_layers):
            linears = nn.ModuleList()
            if i >= self.cb_del_stages - 1:
                jrange = 4 - i
                for j in range(jrange):
                    if cb_inplanes[i + j] != cb_inplanes[i]:
                        layer = nn.Conv2d(cb_inplanes[i + j], cb_inplanes[i], 1)
                    else:
                        layer = nn.Identity()
                    linears.append(layer)
            self.cb_linears.append(linears)

    def init_weights(self):
        for m in self.cb_modules:
            m.init_weights()

    def spatial_interpolate(self, x, H, W):
        if H != x.shape[2] or W != x.shape[3]:
            x = F.interpolate(x, size=(H, W), mode='nearest')
        return x

    def _get_cb_feats(self, feats, tmps):
        cb_feats = []
        Wh, Ww = tmps[0][1:3]
        for i in range(self.num_layers):
            feed = 0
            if i >= self.cb_del_stages - 1:
                jrange = 4 - i
                for j in range(jrange):
                    tmp = self.cb_linears[i][j](feats[j + i])
                    tmp = self.spatial_interpolate(tmp, Wh, Ww)
                    tmp = tmp.permute(0, 2, 3, 1)  # [B, H, W, C]
                    feed += tmp
            cb_feats.append(feed)
            Wh, Ww = Wh // 2, Ww // 2

        return cb_feats

    def forward(self, x):
        outs = []
        for i, module in enumerate(self.cb_modules):
            if i == 0:
                feats, tmps = module(x)
            else:
                feats, tmps = module(x, cb_feats, tmps)
            outs.append(feats)

            if i < len(self.cb_modules) - 1:
                cb_feats = self._get_cb_feats(outs[-1], tmps)

        if len(self.out_indices[0]) == len(self.out_indices[1]) + 1:
            outs[0] = outs[0][1:]

        return tuple(outs)

    def train(self, mode=True):
        super(CBInternImage, self).train(mode)
        for m in self.cb_modules:
            m.train(mode=mode)
        for m in self.cb_linears.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
