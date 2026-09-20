"""Checks the Windows-microphone path (winmic.py serving, voice.WinMic reading) with a synthetic tone as the microphone, so it needs
no audio hardware and no Windows: the helper runs right here. It uses its own token file and port, never the real ones.

  python winmic_test.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

import voice

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 48999


def wait_for(cond, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    checks = {}
    token_file = os.path.join(tempfile.gettempdir(), "winmic_test_token")
    if os.path.exists(token_file):
        os.remove(token_file)
    helper = subprocess.Popen([sys.executable, os.path.join(HERE, "winmic.py"), "--serve", "--device", "tone", "--bind", "127.0.0.1",
                               "--port", str(PORT), "--token-file", token_file], stderr=subprocess.DEVNULL)
    try:
        checks["the helper starts and writes its secret token"] = wait_for(lambda: os.path.exists(token_file) and os.path.getsize(token_file) > 0, 10)
        time.sleep(0.5)

        # a client without the token is turned away
        s = socket.socket()
        s.settimeout(3)
        s.connect(("127.0.0.1", PORT))
        s.sendall(b"not-the-token")
        try:
            data = s.recv(64)
        except OSError:
            data = b""
        s.close()
        checks["a client with the wrong token is refused (nothing is sent to it)"] = data == b""

        mic = voice.WinMic(device="tone", host="127.0.0.1", port=PORT, token_path=token_file)
        mic.open()
        checks["the app connects and audio arrives"] = wait_for(lambda: mic.frames >= 10, 8)
        time.sleep(0.6)
        chunk = mic.read_chunk(100)
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        peak = np.fft.rfftfreq(len(x), 1 / voice.RATE)[np.argmax(np.abs(np.fft.rfft(x)))]
        print(f"  {mic.frames} chunks received; 100 ms chunk: {len(chunk)} bytes, peak {peak:.0f} Hz, RMS {np.sqrt(np.mean(x ** 2)):.3f}, level meter {mic.level:.3f}")
        checks["read_chunk returns exactly 100 ms of the tone (440 Hz, RMS ~0.21)"] = len(chunk) == 3200 and abs(peak - 440) < 25 and 0.15 < float(np.sqrt(np.mean(x ** 2))) < 0.27
        checks["the level meter reads the loudness"] = 0.1 < mic.level < 0.3

        mic.start()                                                 # push-to-talk from the same capture
        time.sleep(1.2)
        rec = mic.stop()
        print(f"  push-to-talk recorded {len(rec) / 32000:.2f} s while held for 1.2 s")
        checks["push-to-talk (start/stop) records about what was held (1.0-1.4 s)"] = 32000 <= len(rec) <= 45000
        checks["...and the always-on stream kept flowing while it did"] = mic.read_chunk(100) != bytes(3200)

        mic.close()
        checks["after close() a read returns None (the ear stops)"] = mic.read_chunk(100) is None
        before = mic.frames
        mic.open()                                                  # the ear turned off and on again (L), or restarted after a fallback
        checks["...and open() again reconnects and audio flows again"] = wait_for(lambda: mic.frames > before + 5, 10)
        mic.close()

        # no helper running: a clear message, and the ear stays alive on silence
        lonely = voice.WinMic(device="tone", host="127.0.0.1", port=PORT + 1, token_path=token_file)
        lonely.open()
        wait_for(lambda: bool(lonely.error), 6)
        print("  with no helper:", lonely.error)
        checks["with no helper running it says to launch go2.bat, and never reports audio"] = "go2.bat" in lonely.error and lonely.frames == 0
        checks["...and the ear gets silence, not a crash"] = lonely.read_chunk(100) == bytes(3200)
        lonely.close()
    finally:
        helper.terminate()
        try:
            helper.wait(5)
        except subprocess.TimeoutExpired:
            helper.kill()
        if os.path.exists(token_file):
            os.remove(token_file)

    checks.update(app_fallback_checks())
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


def app_fallback_checks() -> dict:
    """What go2.py does a few seconds after the ear starts on the Windows mic: live -> say so; silent or absent -> say why and use the
    computer's microphone instead (a deaf ear is worse than the wrong microphone)."""
    try:
        import go2
    except Exception as e:  # noqa: BLE001 - pygame etc. missing: skip these
        print(f"  (skipping the app-level fallback checks: {type(e).__name__}: {e})")
        return {}

    class FakeWin:
        def __init__(self, frames, level, error=""):
            self.frames, self.level, self.error, self.closed = frames, level, error, False

        def close(self):
            self.closed = True

    def run(win):
        app = go2.App.__new__(go2.App)
        app.args = type("A", (), {"mic": "win", "mic_device": "AirPods", "wake_word": "ernest"})()
        app.winmic, app._winmic_check, app.said, app.calls, app.mic = win, 0.0, [], [], "win"
        app.say = lambda text, color=None: app.said.append(text)
        app.stop_listener = lambda: app.calls.append("stop")
        app.start_listener = lambda: app.calls.append("start")
        go2.voice_mod.MicRecorder = lambda: "pc-mic"
        app.update_winmic(1.0)
        return app

    live = run(FakeWin(200, 0.004))
    silent = run(FakeWin(200, 0.00001))
    absent = run(FakeWin(0, 0.0, "can't reach the Windows mic helper at 192.168.160.1:48123 (ConnectionRefusedError): launch go2.bat, which starts it"))
    print("  live   :", live.said[0][:90])
    print("  silent :", silent.said[0][:150])
    print("  absent :", absent.said[0][:150])
    return {
        "a live AirPods mic is reported, and nothing is switched": "live" in live.said[0] and live.calls == [] and live.args.mic == "win",
        "a SILENT AirPods mic is explained (ears / iPhone / Windows) and the ear moves to the computer's mic":
            "SILENT" in silent.said[0] and "iPhone" in silent.said[0] and silent.args.mic == "pc" and silent.calls == ["stop", "start"] and silent.mic == "pc-mic",
        "a missing helper says what to do (launch go2.bat) and the ear moves to the computer's mic":
            "go2.bat" in absent.said[0] and absent.args.mic == "pc" and absent.calls == ["stop", "start"],
        "the silent mic's connection was closed on the way out": silent.winmic is None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
