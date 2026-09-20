"""Windows-side microphone helper. Captures ONE input device, chosen by name (default: the AirPods), and serves 16 kHz mono
PCM16 to the app over a local socket.

Why: the always-on ear runs inside WSL, which only ever sees whichever microphone Windows has as its DEFAULT, and WSL can't start
Windows programs here. go2.bat starts this on Windows first; voice.WinMic (in WSL) connects to it and reads the audio.

  python winmic.py --list                      list the input devices
  python winmic.py --device AirPods --test 3   capture 3 s from that device and print the level (nothing else)
  python winmic.py --serve --device AirPods    what go2.bat runs: waits for the app, then streams the mic to it

Privacy: it serves a LIVE MICROPHONE, so (1) it listens only on the WSL virtual network address, never on Wi-Fi or Ethernet
(if that address can't be found it listens on this PC only, and the app can't reach it); (2) the first thing a client must send is
a secret token this program writes to .winmic_token in this folder (only programs that can read the folder can connect);
(3) it captures only while a client is connected and exits by itself when the app has gone.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import time

import numpy as np

RATE = 16000
PORT = 48123
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, ".winmic_token")
LOG_PATH = os.path.join(HERE, "winmic.log")
API_ORDER = ("Windows WASAPI", "MME", "Windows DirectSound")      # best first; WDM-KS is skipped (exclusive, flaky)
IDLE_NEVER_CONNECTED = 300       # s: exit if the app never connects
IDLE_AFTER_CLIENT = 30           # s: exit this long after the app disconnects (it restarts us with go2.bat)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def devices():
    import sounddevice as sd

    apis = {i: a["name"] for i, a in enumerate(sd.query_hostapis())}
    return [(i, d, apis[d["hostapi"]]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def pick(name: str):
    """The input device whose name contains `name` (case-insensitive), on the best host API available."""
    want = name.lower()
    hits = [(i, d, a) for i, d, a in devices() if want in d["name"].lower() and a in API_ORDER]
    hits.sort(key=lambda h: API_ORDER.index(h[2]))
    return hits[0] if hits else None


def open_stream(idx: int, api: str, callback):
    """16 kHz mono if the device does it (or Windows can convert); otherwise the device's own rate, converted here."""
    import sounddevice as sd

    last = None
    for kw in ({}, {"extra_settings": sd.WasapiSettings(auto_convert=True)} if "WASAPI" in api else None):
        if kw is None:
            continue
        try:
            return sd.InputStream(device=idx, samplerate=RATE, channels=1, dtype="int16", blocksize=1600, callback=callback, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
    info = sd.query_devices(idx)
    rate, ch = int(info["default_samplerate"]), max(1, min(2, info["max_input_channels"]))
    log(f"opening at the device's own {rate} Hz and converting ({last})")

    def convert(indata, frames, t, status):
        x = indata.astype(np.float32).mean(axis=1)
        n = int(len(x) * RATE / rate)
        y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        callback(y.astype(np.int16).reshape(-1, 1), n, t, status)

    return sd.InputStream(device=idx, samplerate=rate, channels=ch, dtype="int16", blocksize=int(rate / 10), callback=convert)


def wsl_adapter_ip() -> str | None:
    """This PC's address on the WSL virtual network (the 'vEthernet (WSL...)' adapter), from ipconfig."""
    try:
        text = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for block in re.split(r"\r?\n\r?\n(?=\S)", text):
        if re.search(r"vEthernet.*WSL", block.splitlines()[0] if block.strip() else "", re.I):
            m = re.search(r"IPv4[^:]*:\s*([\d.]+)", block)
            if m:
                return m.group(1)
    return None


class ToneStream:
    """A synthetic 440 Hz microphone (for tests: no audio hardware needed)."""

    def __init__(self, callback):
        self.cb, self.active, self._n = callback, False, 0

    def __enter__(self):
        import threading

        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.active = False

    def _run(self):
        while self.active:
            t = (np.arange(1600) + self._n) / RATE
            self._n += 1600
            self.cb((0.3 * np.sin(2 * math.pi * 440 * t) * 32767).astype(np.int16).reshape(-1, 1), 1600, None, None)
            time.sleep(0.1)


def make_stream(device: str, callback):
    if device == "tone":
        return ToneStream(callback), "the synthetic tone"
    found = pick(device)
    if found is None:
        raise RuntimeError(f"no input device with '{device}' in its name (run: python winmic.py --list)")
    idx, dev, api = found
    return open_stream(idx, api, callback), f"[{idx}] {dev['name']} via {api}"


def serve(device: str, port: int, bind: str | None, token_path: str = TOKEN_PATH) -> int:
    host = bind or wsl_adapter_ip()
    if not host:
        log("could not find the WSL network adapter: listening on this PC only (127.0.0.1); WSL will not be able to connect")
        host = "127.0.0.1"
    srv = socket.socket()
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):                  # Windows: nobody else may share this port (plain REUSEADDR would allow that)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:                                                       # Linux (tests): reuse is safe there and skips the cool-down after a restart
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
    except OSError as e:
        log(f"port {port} is already in use ({e}): another winmic.py is probably running, leaving it (and its token) alone")
        return 1
    srv.listen(1)
    token = secrets.token_hex(16)                               # written only once we own the port
    with open(token_path, "w", encoding="ascii") as f:
        f.write(token)
    srv.settimeout(1.0)
    log(f"serving the '{device}' microphone on {host}:{port} (token in {token_path})")
    started, last_client = time.time(), None
    while True:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            idle = time.time() - (last_client if last_client else started)
            if idle > (IDLE_AFTER_CLIENT if last_client else IDLE_NEVER_CONNECTED):
                log("no client any more: exiting")
                return 0
            continue
        conn.settimeout(5.0)
        try:
            got = conn.recv(64).decode("ascii", "ignore").strip()
        except OSError:
            got = ""
        if not secrets.compare_digest(got, token):
            log(f"refused a connection from {addr[0]}: wrong token")
            conn.close()
            continue
        conn.settimeout(None)
        log(f"client connected from {addr[0]}")
        try:
            dead: list = []

            def send(indata, frames, t, status):
                if dead:
                    return
                try:
                    conn.sendall(indata.astype(np.int16).tobytes())
                except OSError:                                       # the app went away: stop quietly
                    dead.append(1)

            stream, desc = make_stream(device, send)
        except Exception as e:  # noqa: BLE001
            log(f"could not open the microphone: {e}")
            conn.sendall(b"")
            conn.close()
            last_client = time.time()
            continue
        log(f"capturing {desc}")
        try:
            with stream:
                while getattr(stream, "active", True) and not dead:
                    time.sleep(0.3)
                    try:                                            # has the client hung up?
                        conn.setblocking(False)
                        if conn.recv(1, socket.MSG_PEEK) == b"":
                            break
                    except BlockingIOError:
                        pass
                    except OSError:
                        break
                    finally:
                        conn.setblocking(True)
        except OSError:
            pass
        finally:
            conn.close()
            last_client = time.time()
            log("client gone; microphone released")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--device", default="AirPods", help="part of the input device's name; 'tone' = a synthetic tone for tests")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--bind", default=None, help="address to listen on (default: this PC's WSL network address)")
    ap.add_argument("--token-file", default=TOKEN_PATH, help=argparse.SUPPRESS)
    ap.add_argument("--test", type=float, default=0.0, help="capture this many seconds, print the level, exit")
    a = ap.parse_args()
    if a.list:
        for i, d, api in devices():
            print(f"[{i:2d}] {d['name']}  ({api}, {d['max_input_channels']} ch, {d['default_samplerate']:.0f} Hz)")
        return 0
    if a.serve:
        return serve(a.device, a.port, a.bind, a.token_file)
    grabbed: list = []
    stream, desc = make_stream(a.device, lambda indata, frames, t, status: grabbed.append(indata.copy()))
    print(f"using {desc}", flush=True)
    with stream:
        time.sleep(a.test or 3.0)
    x = np.concatenate(grabbed)[:, 0].astype(np.float32) / 32768.0 if grabbed else np.zeros(1, np.float32)
    print(f"{len(x)} samples, RMS {np.sqrt(np.mean(x ** 2)):.5f}, peak {np.abs(x).max():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
