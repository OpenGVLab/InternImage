import torch
import argparse
import math
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)
parser.add_argument('--ema', action='store_true')
args = parser.parse_args()


def gen_grid(n_heads):
    n_heads = n_heads
    n_points = 9
    points_list = []
    kernel_size = int(math.sqrt(n_points))
    y, x = torch.meshgrid(
        torch.linspace((-kernel_size // 2 + 1), (kernel_size // 2),
                       kernel_size,
                       dtype=torch.float32),
        torch.linspace((-kernel_size // 2 + 1), (kernel_size // 2),
                       kernel_size,
                       dtype=torch.float32))
    points_list.extend([y, x])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, n_heads, 1).permute(1, 0, 2)

    return grid


def convert_to_newop(m):
    new_sd = OrderedDict()
    n_points = 9
    for k, v in m.items():
        new_k = k
        if 'attn' in k:
            new_k = new_k.replace('attn', 'dcn')
            if 'sampling_offsets' in k:
                new_k = new_k.replace('sampling_offsets', 'offset')
            if 'attention_weights' in k:
                new_k = new_k.replace('attention_weights', 'mask')
            if 'value_proj' in k:
                new_k = new_k.replace('value_proj', 'input_proj')
        if 'ema' in k:
            continue
        if ".norm1_k." in k:
            new_k = new_k.replace('.norm1_k.', '.norm1_k.0.')
        if ".norm1_q." in k:
            new_k = new_k.replace('.norm1_q.', '.norm1_q.0.')
        if ".norm1_v." in k:
            new_k = new_k.replace('.norm1_v.', '.norm1_v.0.')
        if ".post_norms." in k:
            new_k = new_k.replace('.bias', '.0.bias')
            new_k = new_k.replace('.weight', '.0.weight')
        if "fc_norm." in k:
            new_k = new_k.replace('fc_norm.', 'fc_norm.0.')

        new_sd[new_k] = v.half()

    return new_sd


model = torch.load(args.filename, map_location=torch.device('cpu'))
new_model = {"model": convert_to_newop(model)}
torch.save(new_model, args.filename.replace(".pth", "_rename.pth"))
