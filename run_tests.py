import os
import torch
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
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


def test_model(model_name, model_tch, model_tiny):
    load_pth(f'models_pth/{model_name}.pth', model_tch)
    load_state_dict(model_tiny, safe_load(f'models_st/{model_name}.safetensors'))

    out_tch = model_tch(torch.ones(1, 3, 32, 32)).detach().numpy()
    out_tiny = model_tiny(Tensor.ones(1, 3, 32, 32)).numpy()
    
    np.testing.assert_allclose(out_tch, out_tiny, atol=1e-5, rtol=1e-5)
    print(f'{model_name} -> OK')


model_tch = RRDBNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
model_tiny = RRDBNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
test_model('RealESRGAN_x4plus', model_tch, model_tiny)

model_tch = SRVGGNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
model_tiny = SRVGGNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
test_model('realesr-animevideov3', model_tch, model_tiny)

model_tch = SRVGGNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_tiny = SRVGGNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
test_model('realesr-general-x4v3', model_tch, model_tiny)
