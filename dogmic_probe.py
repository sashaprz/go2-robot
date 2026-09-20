"""Read-only check of the dog's own microphone. Run it with the dog on and its Wi-Fi joined (dogmic-probe.bat). It NEVER moves the dog.

It switches the dog's audio channel on, records ~44 s, and afterwards runs the recording through the same Whisper model the app
uses, so you see what the dog would have understood. It ALWAYS leaves, next to this script (even if it fails or you close it):
  dogmic_probe_report.txt   everything printed, the noise/speech levels, what Whisper heard and how the phrases matched
  dogmic_recording.wav      what the dog's microphone picked up (16 kHz mono), so it can be listened to

What to do (the window prints each step):
  0-8 s    stay silent (measures the dog's own noise, e.g. its fans and servos while it stands)
  8-22 s   say "ernest, sit down" / "ernest, follow me" / "ernest, stop" a few times, about 1 m from the dog
  22-36 s  the same from about 3 m away
  36-44 s  stay silent again
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import wave

import numpy as np

import voice

HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 44.0
REPORT_PATH = os.path.join(HERE, "dogmic_probe_report.txt")
WAV_PATH = os.path.join(HERE, "dogmic_recording.wav")
STAGES = [(0, "STAY SILENT: the dog is measuring its own noise"),
          (8, 'SAY "ernest, sit down", "ernest, follow me", "ernest, stop" (a few times), about 1 m from the dog'),
          (22, "the same from about 3 m away"), (36, "STAY SILENT again")]
REPORT: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    REPORT.append(text)
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT) + "\n")
    except OSError:
        pass


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def main() -> int:
    key = os.environ.get("UNITREE_AES_128_KEY")
    if not key:
        say("UNITREE_AES_128_KEY isn't set (expected in ~/.dimos.env)")
        return 2
    from dimos.robot.unitree.connection import UnitreeWebRTCConnection

    ip = os.environ.get("ROBOT_IP", "192.168.12.1")
    say(f"probe started {time.strftime('%Y-%m-%d %H:%M:%S')}; connecting to {ip} ...")
    c = None
    for attempt in (1, 2, 3):
        try:
            c = UnitreeWebRTCConnection(ip, aes_128_key=key)
            break
        except Exception as e:  # noqa: BLE001
            timed_out = "Timeout" in type(e).__name__ or "timed out" in str(e)
            say(f"attempt {attempt} of 3 failed: {type(e).__name__}")
            if not timed_out:
                raise
            if attempt < 3:
                say("  Something else is probably connected to the dog. Close the Unitree Go phone app COMPLETELY and any go2.bat / other dog window. Waiting 20 s ...")
                time.sleep(20)
    if c is None:
        say("COULD NOT CONNECT to the dog (a connection problem, not a microphone result). Close the phone app and other dog windows, "
            "make sure the PC is on the dog's Wi-Fi, and try again; if it still fails, power the dog off and on.")
        return 3
    say("connected")

    mic = voice.DogMic()
    mic.open()
    pcm_all = bytearray()
    levels: list[tuple[float, float]] = []                   # (seconds since start, RMS of that 100 ms)
    stop = threading.Event()
    t0 = time.time()

    def reader() -> None:
        while not stop.is_set():
            chunk = mic.read_chunk(100)
            if chunk is None:
                return
            pcm_all.extend(chunk)
            levels.append((time.time() - t0, rms(chunk)))

    async def handle(frame) -> None:
        mic.feed(frame)

    try:
        c.conn.audio.add_track_callback(handle)
        c.loop.call_soon_threadsafe(c.conn.audio.switchAudioChannel, True)
        threading.Thread(target=reader, daemon=True).start()
        say(f"listening for {SECONDS:.0f} s (never moves the dog)")
        shown = set()
        while time.time() - t0 < SECONDS:
            time.sleep(1.0)
            for at, what in STAGES:
                if time.time() - t0 >= at and at not in shown:
                    shown.add(at)
                    say(f"NOW ({at}-s mark): {what}")
            t = time.time() - t0
            recent = [v for s, v in levels if s > t - 1.0]
            say(f"t={t:4.0f}s  audio frames {mic.frames}  ({mic.sample_rate} Hz, {mic.channels} ch)  level now {np.mean(recent) if recent else 0:.4f}  peak {max(recent) if recent else 0:.4f}")
    except BaseException as e:  # noqa: BLE001 - Ctrl-C, a dropped connection, anything: still write what we have
        say(f"STOPPED EARLY: {type(e).__name__}: {e}")
        say(traceback.format_exc(limit=4))
    finally:
        stop.set()
        mic.close()
        finish(mic, bytes(pcm_all), levels)
        try:
            c.stop()
        except Exception:  # noqa: BLE001
            pass
    return 0


def finish(mic, pcm: bytes, levels) -> None:
    say()
    if mic.frames == 0:
        say("RESULT: NO audio arrived from the dog. This dog does not stream its microphone over this connection, so the dog's own mic "
            "can't be used here (the app falls back to the computer's microphone automatically).")
        return
    with wave.open(WAV_PATH, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(voice.RATE)
        w.writeframes(pcm)
    say(f"saved {len(pcm) / 32000:.0f} s of audio to {WAV_PATH}")
    sil = [v for s, v in levels if s < 8 or s > 37]
    speech = [v for s, v in levels if 8.5 < s < 35 and v > 0.0]
    floor = float(np.median(sil)) if sil else 0.0
    loud = float(np.percentile(speech, 90)) if speech else 0.0
    say(f"the dog's own noise floor (silent stages): {floor:.4f} RMS;  loudest speech: {loud:.4f} RMS;  about {20 * np.log10(max(loud, 1e-6) / max(floor, 1e-6)):.0f} dB above the noise")
    say("what Whisper made of it (the same model and phrase matcher the app uses):")
    try:
        stt = voice.LocalWhisperSTT("small.en")
        stt.load()
        seg = voice.UtteranceSegmenter()
        utts = []
        for i in range(0, len(pcm), 3200):
            utts += seg.feed(pcm[i:i + 3200])
        heard = 0
        for u in utts:
            text = stt.transcribe(u)
            action, rest = voice.route_utterance(text)
            intent = voice.parse_command(rest) if action in ("command", "stop") else None
            heard += intent is not None
            say(f"   {len(u) / 32000:4.1f} s  heard {text!r:44s} -> {action}" + (f" ({intent.label})" if intent else ""))
        say(f"{heard} of {len(utts)} utterances came out as a command.")
    except Exception as e:  # noqa: BLE001
        say(f"(couldn't run the speech model on it: {type(e).__name__}: {e})")
    say("RESULT: the dog's microphone streams audio. Check the numbers above; listen to dogmic_recording.wav to judge the quality.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        say("PROBE FAILED BEFORE LISTENING:" + chr(10) + traceback.format_exc(limit=6))
        raise SystemExit(1)
