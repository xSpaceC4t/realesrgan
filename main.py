import numpy as np
from rrdbnet_tch import ResidualDenseBlock as BlockTch
from rrdbnet_tiny import ResidualDenseBlock as BlockTiny
from tinygrad.nn.state import get_state_dict
from safetensors.torch import save_file
from tinygrad.nn.state import safe_load, load_state_dict
import torch
from tinygrad import Tensor

tch = BlockTch()
tiny = BlockTiny()

save_file(tch.state_dict(), 'model.safetensors')
load_state_dict(tiny, safe_load('model.safetensors'))

a = tch.forward(torch.ones(1, 64, 32, 32)).detach().numpy()
b = tiny(Tensor.ones(1, 64, 32, 32)).numpy()
print(np.allclose(a, b))

# model_tch = RRDBNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
# model_tiny = RRDBNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

# print(model_tch.forward(torch.ones(1, 3, 32, 32)))
# print(model_tiny(Tensor.ones(1, 3, 32, 32)).numpy())
