"""TCP delay proxy: adds one-way delay to every segment in both directions,
so a request/response RPC over the proxy pays one injected RTT.

Usage: python3 delay_proxy.py <listen_port> <target_port> <one_way_ms>
"""
import asyncio
import sys


async def pipe(reader, writer, delay):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            await asyncio.sleep(delay)
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_r, client_w, target_port, delay):
    try:
        server_r, server_w = await asyncio.open_connection("127.0.0.1", target_port)
    except OSError:
        client_w.close()
        return
    await asyncio.gather(pipe(client_r, server_w, delay),
                         pipe(server_r, client_w, delay))


async def main():
    listen, target, ms = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    delay = ms / 1000.0
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, target, delay), "127.0.0.1", listen)
    print(f"delay proxy :{listen} -> :{target} one-way {ms} ms", flush=True)
    async with server:
        await server.serve_forever()


asyncio.run(main())
