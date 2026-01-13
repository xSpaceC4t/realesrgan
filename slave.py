import asyncio
import hashlib
import pickle
import numpy as np


async def recv_msg(reader):
    msg_len = await reader.readexactly(4)
    msg_len = int.from_bytes(msg_len, byteorder='little')
    data = await reader.readexactly(msg_len)
    return data


async def main():
    reader, writer = await asyncio.open_connection('127.0.0.1', 8888)

    data = await recv_msg(reader)
    target_chksum = await recv_msg(reader)
    print(data)
    print(target_chksum)

    chksum = hashlib.sha256(data).hexdigest().encode()
    assert chksum == target_chksum

    data = pickle.loads(data)
    print(data)

    writer.close()
    await writer.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
