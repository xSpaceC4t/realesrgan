import os
import torch
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from rrdbnet_tch import RRDBNet as RRDBNetTch
from rrdbnet_tiny import RRDBNet as RRDBNetTiny
from srvgg_tch import SRVGGNetCompact as SRVGGNetTch
from srvgg_tiny import SRVGGNetCompact as SRVGGNetTiny


os.environ['DEBUG'] = '2'
os.environ['CPU'] = '1'


def load_pth(path, model):
    loadnet = torch.load(path, map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)


model_tch = RRDBNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
load_pth('models_pth/RealESRGAN_x4plus.pth', model_tch)

model_tiny = RRDBNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
load_state_dict(model_tiny, safe_load('models_st/RealESRGAN_x4plus.safetensors'))

out_tch = model_tch(torch.ones(1, 3, 32, 32)).detach().numpy()
out_tiny = model_tiny(Tensor.ones(1, 3, 32, 32)).numpy()

np.testing.assert_allclose(out_tch, out_tiny, atol=1e-5, rtol=1e-5)
