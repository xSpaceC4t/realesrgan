

sys.exit()



print(img.shape)
_, _, height, width = img.shape

scale = 4
tile_size = 128 - 20
tile_pad = 10

tiles_x = math.ceil(width / tile_size)
tiles_y = math.ceil(height / tile_size)

@TinyJit
def jit(x):
    return model_tiny(x).numpy()

def get_tile(y, height):
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

# img_in = Tensor(img)
img_in = img.astype(np.float32) / 255.0
img_out = np.zeros([1, 3, height * scale, width * scale])

tiles = []
for y in range(tiles_y):
    y1, y2 = get_tile(y, height)
    assert y2 - y1 == tile_size + tile_pad * 2
    for x in range(tiles_x):
        x1, x2 = get_tile(x, width)
        assert x2 - x1 == tile_size + tile_pad * 2
        tile = {
            'coords': [y1, y2, x1, x2],
            'data': img_in[:, :, y1:y2, x1:x2],
        }
        tiles.append(tile)

        x = np.clip(tile['data'], 0.0, 1.0)
        x = (x * 255.0).round().astype(np.uint8)
        x = x.squeeze()
        x = np.transpose(x[[2, 1, 0], :, :], (1, 2, 0))

        # cv2.imshow('My Image', x)
        # cv2.waitKey(0)  # Waits for key press
        # cv2.destroyAllWindows()

tiles_out = []
for tile_in in tqdm(tiles):
    out = jit(Tensor(tile_in['data']))
    # out = Tensor.rand(1, 3, 512, 512).numpy()
    # out = model_tiny(Tensor(tile_in['data']))
    tile_out = {
        'coords': [y1 * scale, y2 * scale, x1 * scale, x2 * scale],
        'data': out,
    }
    tiles_out.append(tile_out)

    x = np.clip(tile_out['data'], 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    x = x.squeeze()
    x = np.transpose(x[[2, 1, 0], :, :], (1, 2, 0))

    cv2.imshow('My Image', x)
    cv2.waitKey(0)  # Waits for key press
    cv2.destroyAllWindows()

for tile_out in tiles_out:
    y1, y2, x1, x2 = tile_out['coords']
    img_out[:, :, y1:y2, x1:x2] = tile_out['data']

img_out = np.clip(img_out, 0.0, 1.0)
img_out = (img_out * 255.0).round().astype(np.uint8)
img_out = img_out.squeeze()
img_out = np.transpose(img_out[[2, 1, 0], :, :], (1, 2, 0))
print(img_out.shape)

cv2.imwrite('output.png', img_out)
