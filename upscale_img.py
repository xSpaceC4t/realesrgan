import argparse
import cv2
import os
import sys
import queue
import threading
import math
import shutil
import hashlib
import numpy as np
from tinygrad import Tensor, TinyJit
from tqdm import tqdm
from utils import *
from itertools import product
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple


TILE_PAD = 10
SCALE = 4


@TinyJit
def jit(model, x):
    return model(x).realize()


def get_tile(y, height, tile_size):
    y1 = y * tile_size
    y2 = y1 + tile_size
    y2 = min(y2, height)
    if y1 == 0:
        y2 += TILE_PAD * 2
    elif y2 == height:
        y1 = y2 - tile_size - TILE_PAD * 2
    else:
        y1 -= TILE_PAD
        y2 += TILE_PAD
    return y1, y2


@dataclass
class Tile:
    in_coords: tuple
    in_data: np.array
    out_coords: tuple
    out_data: np.array


def load_tiles(img, tile_size):
    tile_size -= TILE_PAD * 2
    _, _, height, width = img.shape
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    tiles = []
    for x, y in product(range(tiles_x), range(tiles_y)):
        y1, y2 = get_tile(y, height, tile_size)
        x1, x2 = get_tile(x, width, tile_size)
        tiles.append(Tile(
            in_coords=(y1, y2, x1, x2),
            in_data=img[:, :, y1:y2, x1:x2],
            out_coords=None,
            out_data=None,
        ))

    return tiles


@dataclass
class Item:
    filename: str
    input_dir: str
    height: int
    width: int
    tiles: list


def load(lq, pq, tile_size, input_dir):
    while not lq.empty():
        filename = lq.get()
        img = load_img(f'{input_dir}/{filename}')
        tiles = load_tiles(img, tile_size)
        pq.put(Item(
            filename=filename,
            input_dir=input_dir,
            height=img.shape[2],
            width=img.shape[3],
            tiles=tiles,
        ))
    pq.put(None)


class TinyModel:
    def __init__(self, name):
        self.model = load_model(name, 'tiny', weights=True)

    def __call__(self, x):
        return jit(self.model, Tensor(x)).numpy()


class TchModel:
    def __init__(self, name):
        self.cpu_proc = os.environ.get('CPU', 0)
        self.model = load_model(name, 'tch', weights=True)
        if not self.cpu_proc:
            self.model = self.model.to('cuda')

    def __call__(self, x):
        x = torch.tensor(x)
        if not cpu_proc:
            x = x.to('cuda')
        out = self.model.forward(x).detach().cpu().numpy()
        return out


def proc(lq, pq, sq, model, total_items, bar_mode='tile'):
    if bar_mode == 'image':
        pbar = tqdm(total=total_items, leave=False)

    while True:
        item = pq.get()
        if item == None:
            break

        if bar_mode == 'image':
            loop = range(len(item.tiles))
        else:
            loop = tqdm(range(len(item.tiles)), desc=f'{item.filename}')

        for i in loop:
            out = model(item.tiles[i].in_data)
            item.tiles[i].out_data = out
        sq.put(item)

        if bar_mode == 'image':
            pbar.update(1)

    sq.put(None)


def save(sq, output_path):
    while True:
        item = sq.get()
        if item == None:
            return
        img = np.zeros([1, 3, item.height * SCALE, item.width * SCALE])
        for tile in item.tiles:
            y1, y2, x1, x2 = tile.in_coords
            img[:, :, y1*SCALE:y2*SCALE, x1*SCALE:x2*SCALE] = tile.out_data
        out_path = f'{output_path}/{item.filename}'
        save_img(out_path, img)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '-i', type=str, default=None)
    parser.add_argument('--outputs', '-o', type=str, default=None)
    choices = ['fast', 'balanced', 'quality']
    parser.add_argument('--model', '-m', choices=choices, default='balanced')
    parser.add_argument('--tile', '-t', type=int, default=128)
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--bar-mode', '-bm', choices=['tile', 'image'], default='tile')
    args = parser.parse_args()

    input_dir = args.inputs
    output_dir = args.outputs

    if not input_dir:
        input_dir = 'inputs'
        print(f'WARNING! Input dir is not specified. Defaulting to: {input_dir}')
    if not output_dir:
        output_dir = 'outputs'
        print(f'WARNING! Output dir is not specified. Defaulting to: {output_dir}')

    if not os.path.exists(input_dir):
        print(f'ERROR! Input dir does not exists!')
        sys.exit()
    if not os.path.isdir(input_dir):
        print(f'ERROR! Input path is not dir!')
        sys.exit()
    if not len(os.listdir(input_dir)):
        print(f'ERROR! Input dir is empty!')
        sys.exit()

    if not os.path.exists(output_dir):
        print(f'WARNING! Output dir does not exists! Creating new: {output_dir}')
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir):
        print(f'ERROR! Output path is not dir!')
        sys.exit()
    if args.force:
        print(f'WARNING! "-f/--force" option is set. Output dir is overwritten')
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    if len(os.listdir(output_dir)):
        print(f'ERROR! Output dir is not empty! Set "-f/--force" to overwrite')
        sys.exit()

    model_map = {
        'fast': 'realesr-animevideov3',
        'balanced': 'realesr-general-x4v3',
        'quality': 'RealESRGAN_x4plus',
    }
    model = model_map[args.model] 

    try:
        import torch
        is_torch_available = True
    except ModuleNotFoundError as err:
        print(f'WARNING! Torch is not installed! Defaulting to: Tinygrad')
        is_torch_available = False

    if os.environ.get('TORCH', 0) and is_torch_available:
        print('INFO! Current backend: Torch')
        model = TchModel(model)
    else:
        print('INFO! Current backend: Tinygrad')
        model = TinyModel(model)

    cpu_proc = os.environ.get('CPU', 0)
    if cpu_proc:
        print('INFO! Using CPU device')
    else:
        print('INFO! Using GPU device')

    print(f'INFO! Current model: {args.model} ({model_map[args.model]})')

    imgs = sorted(os.listdir(input_dir))

    lq = queue.Queue()
    for img in imgs:
        lq.put(img)

    pq = queue.Queue(maxsize=10)
    sq = queue.Queue()

    lt = threading.Thread(target=load, args=(lq, pq, args.tile, input_dir,))
    st = threading.Thread(target=save, args=(sq, output_dir,))

    lt.start()
    st.start()

    print(f'INFO! Progress bar mode: {args.bar_mode}')
    proc(lq, pq, sq, model, len(imgs), args.bar_mode)

    lt.join()
    st.join()

    print('INFO! Processing is done!')
