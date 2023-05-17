import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PlaceHolderEncoder(nn.Module):

    def __init__(self, *args, embed_dims=None, **kwargs):
        super(PlaceHolderEncoder, self).__init__()
        self.embed_dims = embed_dims

    def forward(self, *args, query=None, **kwargs):
        
        return query