"""Runs phone.html in a REAL browser (Microsoft Edge, headless, with a fake beeping microphone and the page's synthetic 'demo' motion) against
the real phone_server.py, and checks what a fake dog app receives: 16 kHz mono int16 audio with the beep in it, and 10 Hz motion updates.

  .venv\\Scripts\\python.exe phone_page_test.py

Checks the page's own JavaScript (microphone capture, the resampling to 16 kHz, the WebSocket, the motion messages). It can NOT check what only a
real iPhone does: Safari's permission prompts, the real gyro/accelerometer signs, the screen locking, the Bluetooth microphone route.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import phone_tls

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                         r"C:\Program Files\Google\Chrome\Application\chrome.exe") if os.path.exists(p)), None)
P = dict(https=19443, http=19080, audio=47923, motion=47924)


async def main() -> int:
    if EDGE is None:
        print("  (no Edge/Chrome found: skipping)")
        return 0
    checks = {}
    tmp = tempfile.mkdtemp(prefix="phone_page_")
    tok_file, ptok_file = os.path.join(tmp, "wsl_token"), os.path.join(tmp, "phone_token")
    phone_tls.ensure_tls()
    server = subprocess.Popen([sys.executable, os.path.join(HERE, "phone_server.py"), "--https-port", str(P["https"]), "--http-port", str(P["http"]),
                               "--audio-port", str(P["audio"]), "--motion-port", str(P["motion"]), "--bind-wsl", "127.0.0.1", "--token-file", tok_file,
                               "--phone-token-file", ptok_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
    edge = None
    try:
        for _ in range(100):
            if os.path.exists(tok_file) and os.path.getsize(tok_file) > 0 and os.path.exists(ptok_file):
                break
            await asyncio.sleep(0.1)
        wsl_token, phone_token = open(tok_file).read().strip(), open(ptok_file).read().strip()
        await asyncio.sleep(0.5)
        ar, aw = await asyncio.open_connection("127.0.0.1", P["audio"])
        aw.write(wsl_token.encode())
        mr, mw = await asyncio.open_connection("127.0.0.1", P["motion"])
        mw.write(wsl_token.encode())
        await aw.drain()
        await mw.drain()

        url = f"https://localhost:{P['https']}/phone?t={phone_token}&auto=1&demo=1"
        edge = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={os.path.join(tmp, 'profile')}",
                                 "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required",
                                 "--ignore-certificate-errors", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio, motion = bytearray(), []

        async def read_audio():
            while True:
                d = await ar.read(65536)
                if not d:
                    return
                audio.extend(d)

        async def read_motion():
            while True:
                line = await mr.readline()
                if not line:
                    return
                try:
                    motion.append(json.loads(line))
                except ValueError:
                    pass

        tasks = [asyncio.create_task(read_audio()), asyncio.create_task(read_motion())]
        t0 = time.time()
        while time.time() - t0 < 25 and (len(audio) < 32000 * 6 or len(motion) < 30):
            await asyncio.sleep(0.5)
        for t in tasks:
            t.cancel()
    finally:
        for p in (edge, server):
            if p is not None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()

    x = np.frombuffer(bytes(audio[: len(audio) - len(audio) % 2]), dtype=np.int16).astype(np.float32) / 32768.0
    if os.environ.get("PHONE_TEST_DUMP"):                                    # keep what the page sent, to look at it
        import wave
        with wave.open(os.environ["PHONE_TEST_DUMP"], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(audio[: len(audio) - len(audio) % 2]))
    secs = len(x) / 16000
    chunks = [x[i:i + 1600] for i in range(0, len(x) - 1600, 1600)]
    rms = [float(np.sqrt(np.mean(c ** 2))) for c in chunks]
    loud = [r for r in rms if r > 0.02]
    hz = []
    for c, r in zip(chunks, rms):
        if r > 0.02:                                                     # dominant frequency of each loud chunk (ignoring rumble below 100 Hz)
            f, m = np.fft.rfftfreq(len(c), 1 / 16000), np.abs(np.fft.rfft(c))
            m[f < 100] = 0
            hz.append(float(f[int(np.argmax(m))]))
    peak_hz = float(np.median(hz)) if hz else 0.0
    mm = [m for m in motion if m.get("type") == "m"]
    yaws = [m["yaw"] for m in mm]
    print(f"  audio: {secs:.1f} s received, {len(loud)} of {len(rms)} 100 ms chunks carry the beep (loudest {max(rms) if rms else 0:.3f} RMS, {peak_hz:.0f} Hz)")
    print(f"  motion: {len(mm)} updates, yaw from {min(yaws) if yaws else 0:+.2f} to {max(yaws) if yaws else 0:+.2f}, walking seen: {sorted({m['walk'] for m in mm})}, hello: {[m.get('ua', '')[:40] for m in motion if m.get('type') == 'hello']}")
    checks["the page runs in a real browser and its audio reaches the app (at least 3 s of 16 kHz PCM)"] = secs >= 3.0
    checks["the fake microphone's 400 Hz beep comes out at 400 Hz (+-15): the capture and the 48 kHz -> 16 kHz resampling are right"] = bool(loud) and abs(peak_hz - 400) < 15
    checks["the loudness is sane (not clipped, not silent)"] = bool(rms) and 0.02 < max(rms) < 0.95
    checks["motion updates arrive at about 10 Hz"] = len(mm) >= 20
    checks["the turning rate varies and is flagged as set up (demo mode)"] = bool(yaws) and (max(yaws) - min(yaws)) > 0.5 and all(m["cal"] for m in mm)
    checks["the page identifies itself (hello with the browser's user agent)"] = any(m.get("type") == "hello" and "Mozilla" in m.get("ua", "") for m in motion)

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
