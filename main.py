import numpy as np
from rrdbnet_tch import ResidualDenseBlock as BlockTch
from rrdbnet_tiny import ResidualDenseBlock as BlockTiny
from rrdbnet_tch import RRDB as RRDBTch
from rrdbnet_tiny import RRDB as RRDBTiny
from rrdbnet_tch import RRDBNet as RRDBNetTch
from rrdbnet_tiny import RRDBNet as RRDBNetTiny
from tinygrad.nn.state import get_state_dict
from safetensors.torch import save_file
from tinygrad.nn.state import safe_load, load_state_dict
import torch
from tinygrad import Tensor

# tch = BlockTch()
# tiny = BlockTiny()
# 
# save_file(tch.state_dict(), 'model.safetensors')
# load_state_dict(tiny, safe_load('model.safetensors'))
# 
# a = tch.forward(torch.ones(1, 64, 32, 32)).detach().numpy()
# b = tiny(Tensor.ones(1, 64, 32, 32)).numpy()
# print(np.allclose(a, b))
# 
# tch = RRDBTch(64)
# tiny = RRDBTiny(64)
# 
# save_file(tch.state_dict(), 'model.safetensors')
# load_state_dict(tiny, safe_load('model.safetensors'))
# 
# a = tch.forward(torch.ones(1, 64, 32, 32)).detach().numpy()
# b = tiny(Tensor.ones(1, 64, 32, 32)).numpy()
# print(np.allclose(a, b))

tch = RRDBNetTch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=2, num_grow_ch=32)
tiny = RRDBNetTiny(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=2, num_grow_ch=32)

save_file(tch.state_dict(), 'model.safetensors')
load_state_dict(tiny, safe_load('model.safetensors'))

a = tch.forward(torch.ones(1, 3, 32, 32)).detach().numpy()
b = tiny(Tensor.ones(1, 3, 32, 32)).numpy()
print(np.allclose(a, b, atol=1e-5, rtol=1e-5))

# np.testing.assert_allclose(tg_res, torch_out.cpu().numpy(), atol=1e-5, rtol=1e-5)
