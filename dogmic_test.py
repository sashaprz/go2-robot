"""Checks the dog-microphone path in voice.py with fake audio frames (no dog needed): 48 kHz stereo frames like the ones
WebRTC delivers must come out as 16 kHz mono PCM16 of the right length and pitch.

  python dogmic_test.py       (needs PyAV: run it inside the DimOS environment, e.g. via WSL)
"""
from __future__ import annotations

import threading
import time

import av
import numpy as np

import voice


def stereo_frame(x: np.ndarray, rate: int = 48000) -> "av.AudioFrame":
    """x: mono float samples -> a packed s16 stereo av.AudioFrame (both channels the same), like an Opus-decoded WebRTC frame."""
    s16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    packed = np.repeat(s16, 2).reshape(1, -1)                     # L R L R ...
    f = av.AudioFrame.from_ndarray(packed, format="s16", layout="stereo")
    f.sample_rate = rate
    return f


def main() -> int:
    checks = {}
    mic = voice.DogMic()
    mic.open()
    t = np.arange(48000) / 48000
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)                       # one second of 440 Hz
    for i in range(0, 48000, 960):                                 # 20 ms Opus-sized frames
        mic.feed(stereo_frame(tone[i:i + 960]))
    got = bytes(mic._buf)                                          # everything the resampler produced
    n = len(got) // 2
    first = mic.read_chunk(100)
    checks["read_chunk returns exactly 100 ms (3200 bytes) of it"] = len(first) == 3200
    x = np.frombuffer(got, dtype=np.int16).astype(np.float32) / 32768.0
    peak = np.fft.rfftfreq(n, 1 / voice.RATE)[np.argmax(np.abs(np.fft.rfft(x)))]
    print(f"  1.0 s of 48 kHz stereo in -> {n} samples of 16 kHz mono out (source {mic.sample_rate} Hz, {mic.channels} channels), peak at {peak:.0f} Hz, level {mic.level:.3f}")
    checks["48 kHz stereo becomes about one second of 16 kHz mono (16000 samples +-5%)"] = abs(n - 16000) < 800
    checks["the pitch is preserved (440 Hz tone comes out at 440 Hz)"] = abs(peak - 440) < 15
    checks["the loudness is about right (a 0.3 amplitude tone has RMS ~0.21)"] = 0.15 < float(np.sqrt(np.mean(x ** 2))) < 0.27
    checks["the frames and rate are recorded (for the on-screen 'no audio' check)"] = mic.frames == 50 and mic.sample_rate == 48000 and mic.channels == 2

    # a quiet dog: the ear must keep going on silence, not stop
    with mic._cv:
        mic._buf.clear()                                          # (drain what is left of the tone first)
    t0 = time.time()
    chunk = mic.read_chunk(100)
    checks["when the dog sends nothing, read_chunk returns 100 ms of silence within ~1 s (the listener stays alive)"] = (
        chunk == bytes(3200) and time.time() - t0 < 1.6)

    # a slow speech-to-text must not build up a backlog
    for i in range(0, 48000, 960):
        for _ in range(20):
            mic.feed(stereo_frame(tone[i:i + 960]))
    checks[f"a backlog is capped at {voice.DogMic.MAX_BUFFER_SECONDS} s (got {len(mic._buf) / 32000:.1f} s)"] = len(mic._buf) <= voice.DogMic.MAX_BUFFER_SECONDS * 32000

    # closing wakes a blocked reader and ends it
    m2, out = voice.DogMic(), []
    th = threading.Thread(target=lambda: out.append(m2.read_chunk(100)))
    th.start()
    time.sleep(0.1)
    m2.close()
    th.join(2)
    checks["close() releases a waiting reader and returns None"] = out == [None]

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
