import torch
from safetensors.torch import save_file
from rrdbnet_tch import RRDBNet
from srvgg_tch import SRVGGNetCompact


def load_pth(path, model):
    loadnet = torch.load(path, map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
load_pth('models_pth/RealESRGAN_x4plus.pth', model)
save_file(model.state_dict(), 'models_st/RealESRGAN_x4plus.safetensors')

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
load_pth('models_pth/realesr-animevideov3.pth', model)
save_file(model.state_dict(), 'models_st/realesr-animevideov3.safetensors')

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
load_pth('models_pth/realesr-general-x4v3.pth', model)
save_file(model.state_dict(), 'models_st/realesr-general-x4v3.safetensors')
