import torch
import torch.nn.functional as F
from torch import Tensor

def generate_square_subsequent_mask(sz: int, condition_len: int = 1, bool_out=False, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

    if condition_len > 1:
        mask[:condition_len,:condition_len] = 1

    if not bool_out:
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0)))
    return mask.to(device=device)


def dequantize_verts(verts, canvas_size: Tensor, add_noise=False):
    """Quantizes vertices and outputs integers with specified n_bits."""
    min_range = -1
    max_range = 1
    range_quantize = canvas_size

    verts = verts.type(torch.float32)
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        verts += torch.rand_like(verts) * range_quantize
    return verts


def quantize_verts(
        verts,
        canvas_size: Tensor):
    """Convert vertices from its original range ([-1,1]) to discrete values in [0, n_bits**2 - 1].
        Args:
            verts: seqlen, 2
    """
    min_range = -1
    max_range = 1
    range_quantize = canvas_size-1

    verts_ratio = (verts - min_range) / (
        max_range - min_range)
    verts_quantize = verts_ratio * range_quantize

    return verts_quantize.type(torch.int32)


def top_k_logits(logits, k):
    """Masks logits such that logits not in top-k are small."""
    if k == 0:
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        k_largest = torch.min(values)
        logits = torch.where(logits < k_largest,
                             torch.ones_like(logits)*-1e9, logits)
        return logits


def top_p_logits(logits, p):
    """Masks logits using nucleus (top-p) sampling."""
    if p == 1:
        return logits
    else:

        seq, dim = logits.shape[1:]
        logits = logits.view(-1, dim)
        sort_indices = torch.argsort(logits, dim=-1, descending=True)
        probs = F.softmax(logits, dim=-1).gather(-1, sort_indices)
        cumprobs = torch.cumsum(probs, dim=-1) - probs

        # The top 1 candidate always will not be masked.
        # This way ensures at least 1 indices will be selected.
        sort_mask = (cumprobs > p).type(logits.dtype)
        batch_indices = torch.repeat_interleave(
            torch.arange(logits.shape[0]).unsqueeze(-1), dim, dim=-1)

        top_p_mask = torch.zeros_like(logits)
        top_p_mask = top_p_mask.scatter_add(-1, sort_indices, sort_mask)

        logits -= top_p_mask * 1e9
        return logits.view(-1, seq, dim)
