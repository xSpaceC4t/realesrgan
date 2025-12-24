import argparse
import cv2
import os
import sys
import queue
import threading
import math
import numpy as np
from tinygrad import Tensor, TinyJit
from tqdm import tqdm
from utils import *


@TinyJit
def jit(model, x):
    return model(x).realize()


def get_tile(y, height, tile_size, tile_pad):
    y1 = y * tile_size
    y2 = y1 + tile_size
    y2 = min(y2, height)
    if y1 == 0:
        y2 += tile_pad * 2
    elif y2 == height:
        y1 = y2 - tile_size - tile_pad * 2
    else:
        y1 -= tile_pad
        y2 += tile_pad
    return y1, y2


def load_tiles(img, tile_size):
    scale = 4
    tile_pad = 10
    tile_size -= tile_pad * 2
    _, _, height, width = img.shape
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    tiles = []
    for y in range(tiles_y):
        y1, y2 = get_tile(y, height, tile_size, tile_pad)
        assert y2 - y1 == tile_size + tile_pad * 2
        for x in range(tiles_x):
            x1, x2 = get_tile(x, width, tile_size, tile_pad)
            assert x2 - x1 == tile_size + tile_pad * 2
            tiles.append({
                'coords': [y1, y2, x1, x2],
                'data': img[:, :, y1:y2, x1:x2],
            })

    return tiles


def load(lq, pq, tile_size):
    while not lq.empty():
        file = lq.get()
        img = load_img(f'inputs/{file}')
        tiles = load_tiles(img, tile_size)
        item = {
            'file': file,
            'height': img.shape[2],
            'width': img.shape[3],
            'tiles': tiles,
        }
        pq.put(item)
    pq.put(None)


def proc(lq, pq, sq, model_name, backend='tiny'):
    model = load_model(model_name, backend, weights=True)
    while True:
        item = pq.get()
        if item == None:
            break
        tiles = []
        for tile in tqdm(item['tiles']):
            if backend == 'tiny':
                out = jit(model, Tensor(tile['data'])).numpy()
            else:
                out = model.forward(torch.tensor(tile['data'])).detach().numpy()
            tiles.append({
                'coords': tile['coords'],
                'data': out,
            })
        sq.put({
            'file': item['file'],
            'height': item['height'] * 4,
            'width': item['width'] * 4,
            'tiles': tiles,
        })
    sq.put(None)


def save(sq):
    while True:
        item = sq.get()
        if item == None:
            return
        print(item['file'])
        img = np.zeros([1, 3, item['height'], item['width']])
        for tile in item['tiles']:
            y1, y2, x1, x2 = tile['coords']
            img[:, :, y1*4:y2*4, x1*4:x2*4] = tile['data']
        save_img(f'outputs/{item["file"]}', img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='inputs')
    parser.add_argument('--output', '-o', type=str, default='outputs')
    # parser.add_argument('--model', '-m', type=str, default='RealESRGAN_x4plus')
    parser.add_argument('--model', '-m', type=str, default='realesr-animevideov3')
    parser.add_argument('--tile', '-t', type=int, default=128)
    parser.add_argument('--backend', '-b', type=str, default='tiny')
    args = parser.parse_args()

    imgs = sorted(os.listdir(args.input))

    lq = queue.Queue()
    for img in imgs:
        lq.put(img)

    pq = queue.Queue(maxsize=10)
    sq = queue.Queue()

    lt = threading.Thread(target=load, args=(lq, pq, args.tile,))
    st = threading.Thread(target=save, args=(sq,))

    lt.start()
    st.start()

    proc(lq, pq, sq, args.model, args.backend)

    lt.join()
    st.join()
