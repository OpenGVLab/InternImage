import torch
import argparse
import math
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

def gen_grid(n_heads):
    n_heads = n_heads
    n_points = 9
    points_list = []
    kernel_size = int(math.sqrt(n_points))
    y, x = torch.meshgrid(
        torch.linspace(
            (-kernel_size // 2 + 1),
            (kernel_size // 2), kernel_size,
            dtype=torch.float32),
        torch.linspace(
            (-kernel_size // 2 + 1),
            (kernel_size // 2), kernel_size,
            dtype=torch.float32))
    points_list.extend([y, x])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, n_heads, 1).permute(1, 0, 2)
    
    return grid

def remove_ab(m):
    new_sd = OrderedDict()
    n_points = 9
    for k, v in m.items():
        if 'alpha_beta' in k:
            ab = v
            ab = ab.repeat(1, n_points)
            h, _ = ab.size()

            offset_b = k.replace('alpha_beta', 'sampling_offsets.bias')
            ob = m[offset_b]

            grid = gen_grid(h)
            grid = grid.reshape(h, -1)

            delta = (ab - 1) * grid
            delta = delta.reshape(-1)

            ob = ob + delta

            new_sd[offset_b] = ob
            continue

        if 'sampling_offsets.bias' in k:
            continue

        new_sd[k] = v

    return new_sd

model = torch.load(args.filename, map_location=torch.device('cpu'))
model = remove_ab(model)
# model['model_ema'] = remove_ab(model['model_ema'])
torch.save(model, args.filename.replace(".pth", "_rmab.pth"))
