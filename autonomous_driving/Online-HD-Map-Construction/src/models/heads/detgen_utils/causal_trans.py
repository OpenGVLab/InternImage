# the causal layer is credited by the https://github.com/alexmt-scale/causal-transformer-decoder
# we made some change to stick with the polygen.
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.utils import build_from_cfg


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.
    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError(
                    "cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    causal_mask=causal_mask,
                    only_last=False,
                )

            return output, cache
        else:
            new_token_cache = []
            for i, mod in enumerate(self.layers):
                output = mod(output, memory,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             causal_mask=causal_mask,
                             only_last=True if cache is not None else False)
                new_token_cache.append(output)

                # use the pre_calculated intermediate parameters.
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)

            if cache is not None:
                new_cache = torch.cat(
                    [cache, torch.stack(new_token_cache, dim=0)], dim=1)
            else:
                new_cache = torch.stack(new_token_cache, dim=0)

            return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, *args, re_zero=True, norm_first=True, map_attn_cfg=None, **kwargs):
        '''
            Args:
                re_zero: If True, alpha scale residuals with zero init.
        '''
        super(CausalTransformerDecoderLayer, self).__init__(*args, **kwargs)

        if re_zero:
            self.res_weight1 = nn.Parameter(torch.FloatTensor([0, ]))
            self.res_weight2 = nn.Parameter(torch.FloatTensor([0, ]))
            self.res_weight3 = nn.Parameter(torch.FloatTensor([0, ]))
        else:
            self.res_weight1 = 1.
            self.res_weight2 = 1.
            self.res_weight3 = 1.

        self.norm_first = norm_first

        self.map_attn = None
        if map_attn_cfg is not None:
            self.map_attn = build_attention(map_attn_cfg)

    def forward(
            self,
            tgt: Tensor,
            memory: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            causal_mask: Optional[Tensor] = None,
            query: Optional[Tensor] = None,
            only_last=False) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
            query is not None model will perform query stream 
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """
        if not self.norm_first:
            raise ValueError(
                "norm_first parameter should be True!")

        if self.training:
            # the official Pytorch implementation
            x = tgt
            if query is not None:
                x = query
            
            x = x + self.res_weight1 * \
                self._sa_block(self.norm1(x), self.norm1(tgt), causal_mask,
                                tgt_key_padding_mask)
            if memory is not None:
                x = x + self.res_weight2 * \
                    self._mha_block(self.norm2(x), memory,
                                    memory_mask, memory_key_padding_mask)
            x = x + self.res_weight3*self._ff_block(self.norm3(x))
            
            return x

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.
        # we follow the pre-LN trans in https://arxiv.org/pdf/2002.04745v1.pdf .

        x = tgt
        if query is not None:
            x = query

        if only_last:
            x = x[-1:]
            
        if causal_mask is not None:
            attn_mask = causal_mask 
            if only_last:
                attn_mask = attn_mask[-1:]   # XXX
        else:
            attn_mask = None
            
        # efficient self attention
        x = x + self.res_weight1 * \
            self._sa_block(self.norm1(x), self.norm1(tgt), attn_mask,
                           tgt_key_padding_mask)

        # encoder-decoder attention
        if memory is not None:
            x = x + self.res_weight2 * \
                self._mha_block(self.norm2(x), memory,
                                memory_mask, memory_key_padding_mask)

        # final feed-forward network
        x = x + self.res_weight3*self._ff_block(self.norm3(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, mem: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, mem, mem,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class PolygenTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, *args, re_zero=True, norm_first=True, **kwargs):
        '''
            Args:
                re_zero: If True, alpha scale residuals with zero init.
        '''
        super(PolygenTransformerEncoderLayer, self).__init__(*args, **kwargs)

        if re_zero:
            self.res_weight1 = nn.Parameter(torch.FloatTensor([0, ]))
            self.res_weight2 = nn.Parameter(torch.FloatTensor([0, ]))
        else:
            self.res_weight1 = 1.
            self.res_weight2 = 1.

        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self.res_weight1*self._sa_block(self.norm1(x), src_mask,
                                                    src_key_padding_mask)
            x = x + self.res_weight2*self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self.res_weight1*self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self.res_weight2*self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask