"""Checks the phone link server with a FAKE phone (a Python WebSocket client that trusts only our own root certificate) and fake dog-app
clients on the WSL-facing ports. Runs on Windows with the project .venv (it needs `websockets`):

  .venv\\Scripts\\python.exe phone_test.py

It uses its own ports and token files, never the real ones. What it can't check: the iPhone itself (installing the root certificate, Safari's
permission prompts, the motion sensors, the screen locking).
"""
from __future__ import annotations

import asyncio
import json
import os
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

import phone_tls

HERE = os.path.dirname(os.path.abspath(__file__))
P = dict(https=18443, http=18080, audio=48923, motion=48924)


async def app_client(port: int, token: str):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(token.encode("ascii"))
    await w.drain()
    return r, w


async def read_exact(r, n: int, timeout: float = 5.0) -> bytes:
    return await asyncio.wait_for(r.readexactly(n), timeout)


async def main() -> int:
    checks = {}
    tmp = tempfile.mkdtemp(prefix="phone_test_")
    tok_file, ptok_file = os.path.join(tmp, "wsl_token"), os.path.join(tmp, "phone_token")
    tls = phone_tls.ensure_tls()
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "phone_server.py"), "--https-port", str(P["https"]), "--http-port", str(P["http"]),
                             "--audio-port", str(P["audio"]), "--motion-port", str(P["motion"]), "--bind-wsl", "127.0.0.1", "--token-file", tok_file,
                             "--phone-token-file", ptok_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
    try:
        for _ in range(100):
            if os.path.exists(tok_file) and os.path.getsize(tok_file) > 0 and os.path.exists(ptok_file):
                break
            await asyncio.sleep(0.1)
        checks["the server starts and writes both secrets"] = os.path.exists(tok_file) and os.path.exists(ptok_file)
        wsl_token, phone_token = open(tok_file).read().strip(), open(ptok_file).read().strip()
        await asyncio.sleep(0.5)

        ctx = ssl.create_default_context(cafile=tls["ca_pem"])              # trusts ONLY our own root, like the iPhone will
        base = f"https://localhost:{P['https']}"

        def get(url, use_ctx=ctx):
            try:
                with urllib.request.urlopen(url, context=use_ctx, timeout=5) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, b""
            except Exception as e:  # noqa: BLE001
                return None, str(e).encode()

        st, body = get(f"{base}/phone?t={phone_token}")
        checks["the page loads over HTTPS, verified against our root, with the secret"] = st == 200 and b"Go2 phone link" in body
        checks["the page is refused without the secret"] = get(f"{base}/phone")[0] == 403 and get(f"{base}/phone?t=wrong")[0] == 403
        st, _ = get(f"{base}/ping", ssl.create_default_context())
        checks["a client that does NOT trust our root can't connect (so it really is our certificate doing the work)"] = st is None
        st, ca = get(f"http://localhost:{P['http']}/ca.cer")
        checks["the setup page and the public root certificate are served over plain HTTP"] = st == 200 and ca[:1] == b"\x30"
        st, page = get(f"http://localhost:{P['http']}/")
        checks["...and the setup page never contains the secret"] = st == 200 and phone_token.encode() not in page and b"Certificate Trust Settings" in page

        try:
            async with connect(f"wss://localhost:{P['https']}/ws?t=wrong", ssl=ctx):
                rejected = False
        except InvalidStatus as e:
            rejected = e.response.status_code == 403
        except Exception:  # noqa: BLE001
            rejected = True
        checks["a WebSocket with the wrong secret is refused"] = rejected

        # the dog app (fake) connects to both WSL-facing ports
        ar, aw = await app_client(P["audio"], wsl_token)
        mr, mw = await app_client(P["motion"], wsl_token)
        bad_r, bad_w = await app_client(P["audio"], "not-the-token")
        await asyncio.sleep(0.3)
        checks["an app client with the wrong token is dropped"] = (await bad_r.read(10)) == b""
        bad_w.close()

        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000) * 32767).astype(np.int16)
        async with connect(f"wss://localhost:{P['https']}/ws?t={phone_token}", ssl=ctx) as phone:
            await phone.send(json.dumps({"type": "hello", "ua": "fake iPhone", "sign": 1}))
            hello = json.loads((await asyncio.wait_for(mr.readline(), 5)).decode())
            checks["the phone's hello reaches the app as a motion-channel line"] = hello.get("type") == "hello" and hello.get("ua") == "fake iPhone" and "st" in hello

            for _ in range(3):
                await phone.send(tone.tobytes())
            got = await read_exact(ar, 3 * 3200)
            checks["audio from the phone reaches the app byte-for-byte (3 x 100 ms)"] = got == tone.tobytes() * 3

            await phone.send(json.dumps({"type": "m", "t": 123, "yaw": -0.42, "walk": True, "en": 2.1, "hdg": 90.0, "cal": True}))
            m = json.loads((await asyncio.wait_for(mr.readline(), 5)).decode())
            checks["a motion update arrives as JSON with yaw / walking / calibrated intact"] = (
                m["type"] == "m" and m["yaw"] == -0.42 and m["walk"] is True and m["cal"] is True and abs(m["st"] - time.time()) < 5)

            status = None
            for _ in range(5):
                msg = json.loads(await asyncio.wait_for(phone.recv(), 5))
                if msg.get("type") == "status":
                    status = msg
                    break
            checks["the phone is told the dog app is listening"] = bool(status and status["app"] is True)

            async with connect(f"wss://localhost:{P['https']}/ws?t={phone_token}", ssl=ctx) as phone2:
                await phone2.send(tone.tobytes())
                got2 = await read_exact(ar, 3200)
                await asyncio.sleep(0.3)
                replaced = False
                for _ in range(6):                                  # (routine status messages may still be in flight: read until it closes)
                    try:
                        await asyncio.wait_for(phone.recv(), 2)
                    except Exception:  # noqa: BLE001 - closed
                        replaced = True
                        break
            checks["a second phone replaces the first (one phone at a time)"] = got2 == tone.tobytes() and replaced

        aw.close()
        mw.close()
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
