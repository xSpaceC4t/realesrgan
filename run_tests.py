import torch
from tinygrad import Tensor
import numpy as np
from utils import *


if __name__ == '__main__':
    names = ['RealESRGAN_x4plus', 'realesr-general-x4v3', 'realesr-animevideov3']

    for name in names:
        tch = load_model(name, 'tch', weights=True) 
        tiny = load_model(name, 'tiny', weights=True) 

        out_tch = tch(torch.ones(1, 3, 32, 32)).detach().numpy()
        out_tiny = tiny(Tensor.ones(1, 3, 32, 32)).numpy()

        np.testing.assert_allclose(out_tch, out_tiny, atol=1e-5, rtol=1e-5)
        print(f'{name} -> OK')
