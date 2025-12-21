import os
import torch
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tqdm import tqdm
from rrdbnet_tch import RRDBNet as RRDBNetTch
from rrdbnet_tiny import RRDBNet as RRDBNetTiny
from srvgg_tch import SRVGGNetCompact as SRVGGNetTch
from srvgg_tiny import SRVGGNetCompact as SRVGGNetTiny


def load_pth(path, model):
    loadnet = torch.load(path, map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)


model_tch = SRVGGNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_tiny = SRVGGNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
# test_model('realesr-general-x4v3', model_tch, model_tiny)

tile_size = 128
num_samples = 50

for _ in tqdm(range(num_samples), desc='Torch'):
    x = torch.rand(1, 3, tile_size, tile_size)
    out = model_tch.forward(x)

@TinyJit
def jit(x):
    return model_tiny(x).realize()

for _ in tqdm(range(num_samples), desc='Tinygrad'):
    x = Tensor.rand(1, 3, tile_size, tile_size)
    jit(x)
