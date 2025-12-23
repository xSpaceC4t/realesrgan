import argparse
import cv2
import numpy as np
from tinygrad import Tensor
from utils import *


def load_img(path):
    img = cv2.imread(path)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0
    return img


def save_img(path, img):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    img = img.squeeze()
    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str, default='output.png')
    parser.add_argument('--model', '-m', type=str, default='RealESRGAN_x4plus')
    args = parser.parse_args()

    model = load_model(args.model, 'tiny', weights=True)

    img = load_img(args.input)
    out = model(Tensor(img)).numpy()
    save_img(args.output, out)
