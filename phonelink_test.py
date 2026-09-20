"""Checks the dog app's side of the phone link (phonelink.PhoneLink, voice.PhoneMic, voice.SwitchMic) against the REAL phone_server.py
with a fake phone. Runs on Windows with the project .venv (needs `websockets`):

  .venv\\Scripts\\python.exe phonelink_test.py

Uses its own ports and token files. What it can't check: the real iPhone.
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

import numpy as np
from websockets.asyncio.client import connect

import phone_tls
import phonelink
import voice

HERE = os.path.dirname(os.path.abspath(__file__))
P = dict(https=17443, http=17080, audio=47823, motion=47824)


class FakeMic:
    """A stand-in for the computer's microphone: paced like a real one, and it says which one it is by its sample value."""

    def __init__(self, value: int):
        self.value, self.opened, self.closed = value, False, False

    def open(self):
        self.opened = True

    def read_chunk(self, ms=100):
        time.sleep(ms / 1000 * 0.2)                       # (faster than real time, to keep the test short)
        return np.full(voice.RATE * ms // 1000, self.value, dtype=np.int16).tobytes()

    def start(self):
        self.recording = True

    def stop(self):
        return np.full(1600, self.value, dtype=np.int16).tobytes()

    def close(self):
        self.closed = True


async def wait_for(cond, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.05)               # (never block the event loop: the fake phone's socket needs it to send)
    return False


async def main() -> int:
    checks = {}
    tmp = tempfile.mkdtemp(prefix="phonelink_test_")
    tok_file, ptok_file = os.path.join(tmp, "wsl_token"), os.path.join(tmp, "phone_token")
    tls = phone_tls.ensure_tls()
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "phone_server.py"), "--https-port", str(P["https"]), "--http-port", str(P["http"]),
                             "--audio-port", str(P["audio"]), "--motion-port", str(P["motion"]), "--bind-wsl", "127.0.0.1", "--token-file", tok_file,
                             "--phone-token-file", ptok_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
    link = pmic = smic = None
    try:
        for _ in range(100):
            if os.path.exists(tok_file) and os.path.getsize(tok_file) > 0 and os.path.exists(ptok_file):
                break
            await asyncio.sleep(0.1)
        phone_token = open(ptok_file).read().strip()
        await asyncio.sleep(0.5)
        ctx = ssl.create_default_context(cafile=tls["ca_pem"])

        # ---- PhoneLink.apply / state, no sockets ----
        pl = phonelink.PhoneLink(token_path=os.path.join(tmp, "none"))
        checks["before any update the phone is not live and gives no state"] = (not pl.live) and pl.state() is None
        pl.apply({"type": "hello", "ua": "Mozilla iPhone"})
        pl.apply({"type": "m", "yaw": 0.4, "walk": True, "cal": False, "en": 2.0, "hdg": None}, now=time.time())
        st = pl.state()
        checks["a motion update becomes a follow.PhoneState (yaw, walking, calibrated, fresh)"] = (
            st is not None and abs(st.yaw_rate - 0.4) < 1e-9 and st.walking and not st.calibrated and st.fresh and pl.ua.startswith("Mozilla"))
        pl.apply({"type": "m", "yaw": "garbage"}, now=time.time())
        checks["a malformed update is ignored"] = abs(pl.yaw_rate - 0.4) < 1e-9
        pl.apply({"type": "m", "yaw": 0.0, "walk": False}, now=time.time() - 5)
        checks["an old update is not 'live' (the controller falls back to the camera alone)"] = pl.state() is None

        # ---- the real server, a fake phone, and the app's clients ----
        link = phonelink.PhoneLink(host="127.0.0.1", port=P["motion"], token_path=tok_file)
        link.start()
        pmic = voice.PhoneMic(host="127.0.0.1", port=P["audio"], token_path=tok_file)
        pmic.open()
        checks["the app connects to the phone server's motion port"] = await wait_for(lambda: link.connected)
        checks["the phone mic client connects to the audio port"] = await wait_for(lambda: pmic.connected)

        smic = voice.SwitchMic(pmic, ear=FakeMic(100), ptt=FakeMic(100))
        chunk = smic.read_chunk()
        checks["before the phone streams, the ear uses the computer's microphone"] = smic.source == "computer" and np.frombuffer(chunk, np.int16)[0] == 100

        tone = np.full(1600, 5000, dtype=np.int16).tobytes()
        async with connect(f"wss://localhost:{P['https']}/ws?t={phone_token}", ssl=ctx) as phone:
            await phone.send(json.dumps({"type": "hello", "ua": "fake iPhone"}))
            await phone.send(json.dumps({"type": "m", "t": 1, "yaw": -0.3, "walk": True, "en": 2.0, "hdg": None, "cal": True}))
            checks["the phone's motion reaches PhoneLink through the server"] = await wait_for(lambda: link.live and link.ua == "fake iPhone")
            s = link.state()
            checks["...with yaw / walking / calibrated intact"] = s is not None and abs(s.yaw_rate + 0.3) < 1e-9 and s.walking and s.calibrated

            for _ in range(8):
                await phone.send(tone)
                await asyncio.sleep(0.02)
            checks["phone audio arrives in PhoneMic (frames counted, level > 0)"] = await wait_for(lambda: pmic.frames >= 8 and pmic.level > 0.001)
            got = None
            for _ in range(10):
                c = smic.read_chunk()
                if np.frombuffer(c, np.int16)[0] == 5000:
                    got = c
                    break
                await phone.send(tone)
            checks["while the phone streams, the ear reads the PHONE's audio"] = got is not None and smic.source == "phone"

            await phone.send(json.dumps({"type": "m", "t": 2, "yaw": 0.0, "walk": False, "en": 0.1, "hdg": None, "cal": True}))
            checks["walking -> standing is seen"] = await wait_for(lambda: link.state() is not None and not link.state().walking)

            smic.start()
            await phone.send(tone)
            await asyncio.sleep(0.3)
            rec = smic.stop()
            checks["push-to-talk while the phone streams records the phone's audio"] = len(rec) >= 3200 and np.frombuffer(rec, np.int16)[0] == 5000

        checks["when the phone goes away, its audio stops counting as live after a moment"] = await wait_for(lambda: not smic.live, timeout=4.0)
        chunk = smic.read_chunk()
        checks["...and the ear is back on the computer's microphone by itself"] = np.frombuffer(chunk, np.int16)[0] == 100
        smic.start()
        rec = smic.stop()
        checks["...and push-to-talk uses the computer's microphone again"] = np.frombuffer(rec, np.int16)[0] == 100
        checks["the phone's motion also stops being live (the follower goes back to the camera alone)"] = await wait_for(lambda: link.state() is None, timeout=4.0)

        # ---- the ear's queue and the two speech models (no sockets) ----
        class OneShotSeg:                                     # every chunk it is fed is 'an utterance'
            def feed(self, pcm):
                return [pcm] if pcm[0] else []                # (silence is not an utterance)

        class SlowSTT:
            model_name = "slow"

            def __init__(self):
                self.done = []

            def transcribe(self, pcm):
                time.sleep(0.25)
                self.done.append(pcm[:1])
                return "x"

        class OneMic:
            def __init__(self):
                self.n = 0

            def open(self):
                pass

            def read_chunk(self, ms=100):
                time.sleep(0.01)
                self.n += 1
                return bytes([self.n % 250 + 1]) * 3200 if self.n <= 8 else bytes(3200)

            def close(self):
                pass

        stt = SlowSTT()
        got = []
        ear = voice.AlwaysListener(stt, lambda t: got.append(t), mic=OneMic(), segmenter=OneShotSeg())
        ear.start()
        await asyncio.sleep(2.5)
        ear.stop()
        checks["a backlog of speech drops the OLDEST, so the newest sentence is the one that gets through (8 said fast, slow model)"] = (
            stt.done[-1:] == [bytes([9])] and len(stt.done) < 8 and bytes([2]) not in stt.done[1:])
        checks["the ear reports how long the last sentence took (latency and speech-to-text time)"] = ear.last_stt >= 0.2 and ear.last_latency >= ear.last_stt
        stt2, got2 = SlowSTT(), []
        ear2 = voice.AlwaysListener(stt2, lambda t: got2.append(t), mic=OneMic(), segmenter=OneShotSeg())
        ear2.stale_after = 0.05
        ear2.start()
        await asyncio.sleep(1.5)
        ear2.stop()
        checks["speech that waited too long is dropped, not acted on late (stale_after)"] = len(stt2.done) < 4

        class Tag:
            def __init__(self, name):
                self.model_name = name

            def transcribe(self, pcm):
                return self.model_name

        flag = {"fast": False}
        sw = voice.SwitchSTT(Tag("base.en"), Tag("small.en"), lambda: flag["fast"])
        a = sw.transcribe(b"")
        flag["fast"] = True
        b = sw.transcribe(b"")
        checks["the quick model is used while the phone mic streams, the accurate one otherwise"] = (a, b) == ("small.en", "base.en")
    finally:
        for c in (link, smic):
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass
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
