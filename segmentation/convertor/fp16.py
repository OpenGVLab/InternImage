import torch
import argparse
import math
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)
args = parser.parse_args()


def convert_fl16(m):
    new_sd = OrderedDict()
    for k, v in m.items():
        new_k = k
        new_sd[new_k] = v.half()
    return new_sd


model = torch.load(args.filename, map_location=torch.device('cpu'))['state_dict']
new_model = {"state_dict": convert_fl16(model)}
torch.save(new_model, args.filename.replace(".pth", "_fp16.pth"))
