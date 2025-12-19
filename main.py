import numpy as np
from rrdbnet_tch import ResidualDenseBlock as BlockTch
from rrdbnet_tiny import ResidualDenseBlock as BlockTiny
from tinygrad import nn, Tensor
import torch

tch = BlockTch()
tiny = BlockTiny()

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

for layer_name in layers:
    tch_layer = getattr(tch, layer_name)
    tiny_layer = getattr(tiny, layer_name)

    tiny_layer.weight = Tensor(tch_layer.weight.detach().numpy())
    if hasattr(tiny_layer, 'bias') and tch_layer.bias is not None:
        tiny_layer.bias = Tensor(tch_layer.bias.detach().numpy())

shape = (1, 64, 32, 32)
a = tch.forward(torch.ones(*shape)).detach().numpy()
b = tiny(Tensor.ones(*shape)).numpy()
print(np.allclose(a, b))
