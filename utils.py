import torch
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict


def load_pth(path, model):
    loadnet = torch.load(path, map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)


def load_model(name, backend, weights=False):
    if backend == 'tiny':
        from rrdbnet_tiny import RRDBNet
        from srvgg_tiny import SRVGGNetCompact
    elif backend == 'tch':
        from rrdbnet_tch import RRDBNet
        from srvgg_tch import SRVGGNetCompact
    else:
        print('error: invalid backend name')
        return 

    if name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    elif name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    elif name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    else:
        print('error: invalid model name')
        return

    if backend == 'tiny':
        load_state_dict(model, safe_load(f'models_st/{name}.safetensors'))
    elif backend == 'tch':
        load_pth(f'models_pth/{name}.pth', model)

    return model
