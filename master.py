import asyncio
import pickle
import hashlib
import os
import numpy as np


class Processor:
    def __init__(self):
        self.lq = asyncio.Queue()
        self.pq = asyncio.Queue()
        self.sq = asyncio.Queue()

    def start(self):
        pass

    async def load(self, imgs):
        while not lq.empty():
            file = lq.get()
            img = load_img(f'{input_path}/{file}')
            tiles = load_tiles(img, tile_size)
            item = {
                'file': file,
                'height': img.shape[2],
                'width': img.shape[3],
                'tiles': tiles,
            }
            pq.put(item)
        pq.put(None)

    async def save(self):
        pass


async def send_msg(writer, msg):
    msg_len = len(msg).to_bytes(4, byteorder='little')
    writer.write(msg_len + msg)
    await writer.drain()


async def handle_conn(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"Client connected from {addr}")

    data = np.random.rand(1, 3, 128, 128)
    print(data)
    dump = pickle.dumps(data)
    chksum = hashlib.sha256(dump).hexdigest().encode()
    print(chksum)

    await send_msg(writer, dump)
    await send_msg(writer, chksum)

    writer.close()
    await writer.wait_closed()


async def main():
    imgs = sorted(os.listdir('inputs'))
    print(imgs)

    lt = asyncio.create_task(load(imgs))
    st = asyncio.create_task(save())

    server = await asyncio.start_server(handle_conn, '127.0.0.1', 8888)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
