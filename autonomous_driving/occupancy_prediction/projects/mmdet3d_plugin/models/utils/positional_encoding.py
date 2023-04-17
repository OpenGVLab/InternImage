import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 height_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.height_embed = nn.Embedding(height_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.height_num_embed = height_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        l, h, w = mask.shape[-3:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        z = torch.arange(l, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        z_embed = self.height_embed(z)
        pos = torch.cat(
            (x_embed.unsqueeze(0).unsqueeze(0).repeat(l, h, 1, 1),
             y_embed.unsqueeze(1).unsqueeze(0).repeat(l, 1, w, 1),
             z_embed.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1)),dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(mask.shape[0],1, 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        repr_str += f'height_num_embed={self.height_num_embed})'
        return repr_str
