"""Push-to-talk voice commands for go2.py.

Microphone (WSLg PulseAudio, via libpulse-simple: no apt packages needed) -> speech-to-text -> phrase matcher ->
an Intent that go2.py turns into the same actions the keys trigger.

Two speech-to-text backends:
  * LocalWhisperSTT (default): faster-whisper on CPU. Fully offline, so it works on the dog's own Wi-Fi. Audio never
    leaves the machine. Fetch the model once while online: go2.bat --fetch-model
  * ElevenLabsSTT (--stt elevenlabs): cloud; needs internet and ELEVENLABS_API_KEY in ~/.dimos.env
    (see set-elevenlabs-key.bat). Audio is sent only while the push-to-talk key is held.
"""
from __future__ import annotations

import ctypes
import io
import os
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass

RATE = 16000                 # 16 kHz mono s16 is what the STT wants
MAX_SECONDS = 15             # auto-stop a stuck key
MIN_SECONDS = 0.30           # ignore accidental taps
STT_URL = os.environ.get("ELEVENLABS_STT_URL", "https://api.elevenlabs.io/v1/speech-to-text")
STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2")
STT_LANG = os.environ.get("ELEVENLABS_STT_LANG", "en")   # known language = better/faster on short commands


class VoiceError(Exception):
    """A problem with a friendly message, safe to show in the window."""


# ---- microphone -------------------------------------------------------------------------------------------
class _SampleSpec(ctypes.Structure):
    _fields_ = [("format", ctypes.c_int), ("rate", ctypes.c_uint32), ("channels", ctypes.c_uint8)]


class MicRecorder:
    """Records 16 kHz mono PCM16 from the default PulseAudio source until stop()."""

    def __init__(self):
        self._pa = None
        self._stream = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: list[bytes] = []

    def _lib(self):
        if self._pa is None:
            try:
                import sys
                # macOS uses .dylib with full path, Linux uses .so
                if sys.platform == "darwin":
                    lib_name = "/opt/homebrew/lib/libpulse-simple.0.dylib"
                else:
                    lib_name = "libpulse-simple.so.0"
                pa = ctypes.CDLL(lib_name)
            except OSError as e:
                raise VoiceError(f"PulseAudio library not found: {e}") from e
            pa.pa_simple_new.restype = ctypes.c_void_p
            pa.pa_simple_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                         ctypes.c_char_p, ctypes.POINTER(_SampleSpec), ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            pa.pa_simple_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
            pa.pa_simple_free.argtypes = [ctypes.c_void_p]
            pa.pa_strerror.restype = ctypes.c_char_p
            self._pa = pa
        return self._pa

    def open(self) -> None:
        """Open the PulseAudio record stream (no reading yet)."""
        pa = self._lib()
        err = ctypes.c_int(0)
        spec = _SampleSpec(3, RATE, 1)  # 3 = PA_SAMPLE_S16LE
        stream = pa.pa_simple_new(None, b"go2-voice", 2, None, b"go2 voice",  # 2 = PA_STREAM_RECORD
                                  ctypes.byref(spec), None, None, ctypes.byref(err))
        if not stream:
            raise VoiceError("can't open the microphone: " + pa.pa_strerror(err.value).decode()
                             + " (is WSLg running? Windows mic access allowed?)")
        self._stream = stream

    def start(self) -> None:
        self.open()
        self._frames = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        pa, chunk = self._pa, RATE // 10 * 2            # 100 ms of s16
        buf = ctypes.create_string_buffer(chunk)
        err = ctypes.c_int(0)
        for _ in range(MAX_SECONDS * 10):
            if self._stop.is_set() or pa.pa_simple_read(self._stream, buf, chunk, ctypes.byref(err)) < 0:
                break
            self._frames.append(buf.raw)

    def stop(self) -> bytes:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._stream:
            self._pa.pa_simple_free(self._stream)
            self._stream = None
        return b"".join(self._frames)


class MicStream(MicRecorder):
    """A continuously-open microphone: read_chunk() blocks for ~100 ms and returns that much PCM16 audio."""

    def read_chunk(self, ms: int = 100) -> bytes | None:
        n = RATE * ms // 1000 * 2
        buf = ctypes.create_string_buffer(n)
        err = ctypes.c_int(0)
        if self._stream is None or self._pa.pa_simple_read(self._stream, buf, n, ctypes.byref(err)) < 0:
            return None
        return buf.raw

    def close(self) -> None:
        if self._stream:
            self._pa.pa_simple_free(self._stream)
            self._stream = None


class DogMic:
    """The dog's OWN microphone, streamed over the same WebRTC link (the connection's audio channel must be switched on).

    Same interface as MicStream (open / read_chunk / close), so AlwaysListener can use it instead of the computer's mic.
    feed() takes the audio frames the connection delivers (av.AudioFrame, normally 48 kHz stereo) and turns them into the
    16 kHz mono PCM16 the speech-to-text wants. read_chunk() never blocks forever: if the dog goes quiet it returns silence,
    so the ear stays alive (the app watches `frames` to notice a dog that never sends audio at all)."""

    MAX_BUFFER_SECONDS = 10                     # never let a backlog build up if speech-to-text falls behind

    def __init__(self):
        self._buf = bytearray()
        self._cv = threading.Condition()
        self._closed = False
        self._resampler = None
        self.frames = 0                         # audio frames received so far
        self.last_frame_at = 0.0
        self.level = 0.0                        # smoothed loudness, 0..1 (for the on-screen meter)
        self.sample_rate = 0
        self.channels = 0
        self._rec: bytearray | None = None      # push-to-talk recording, when active

    def open(self) -> None:
        self._closed = False

    def feed(self, frame) -> None:
        import av
        import numpy as np

        if self._resampler is None:
            self._resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
            self.sample_rate, self.channels = frame.sample_rate, len(frame.layout.channels)
        out = self._resampler.resample(frame)
        out = out if isinstance(out, list) else [out]
        pcm = b"".join(f.to_ndarray().tobytes() for f in out if f is not None)
        if not pcm:
            return
        self._push(pcm)

    def _push(self, pcm: bytes) -> None:
        """16 kHz mono PCM16 in: buffer it, track the level, and (push-to-talk) record it."""
        import numpy as np

        x = np.frombuffer(pcm, dtype=np.int16)
        with self._cv:
            self.frames += 1
            self.last_frame_at = time.time()
            self.level = 0.9 * self.level + 0.1 * float(np.sqrt(np.mean((x.astype(np.float32) / 32768.0) ** 2)))
            self._buf += pcm
            cap = self.MAX_BUFFER_SECONDS * RATE * 2
            if len(self._buf) > cap:
                del self._buf[: len(self._buf) - cap]
            if self._rec is not None:
                self._rec += pcm
            self._cv.notify_all()

    def read_chunk(self, ms: int = 100) -> bytes | None:
        n = RATE * ms // 1000 * 2
        with self._cv:
            self._cv.wait_for(lambda: len(self._buf) >= n or self._closed, timeout=1.0)
            if self._closed:
                return None
            if len(self._buf) < n:
                return bytes(n)                 # nothing arrived for a second: silence, so the listener carries on
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
            return chunk

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


def _windows_host_ip() -> str:
    """This PC's address as seen from WSL2 (its NAT gateway), from the routing table. GO2_WIN_HOST overrides."""
    if os.environ.get("GO2_WIN_HOST"):
        return os.environ["GO2_WIN_HOST"]
    try:
        with open("/proc/net/route") as f:
            for line in f.read().splitlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":
                    g = int(parts[2], 16)
                    return ".".join(str((g >> (8 * i)) & 255) for i in range(4))
    except OSError:
        pass
    return "127.0.0.1"


class WinMic(DogMic):
    """A microphone captured on the WINDOWS side by winmic.py (go2.bat starts it) and read here over a local socket, chosen by NAME
    (for example the AirPods) instead of whatever Windows has as its default. WSL can't start Windows programs here, so the helper
    is started by go2.bat and this connects to it.

    One capture serves both the always-on ear (read_chunk) and push-to-talk (start / stop, like MicRecorder). `error` says what is
    wrong when nothing arrives, `frames` counts chunks, `level` is the smoothed loudness (an AirPods mic that is not really connected
    sits at ~0.00001)."""

    def __init__(self, device: str = "AirPods", host: str | None = None, port: int = 48123, token_path: str | None = None):
        super().__init__()
        self.device = device
        self.host = host or _windows_host_ip()
        self.port = port
        self.token_path = token_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".winmic_token")
        self.error = ""
        self.connected = False
        self._thread: threading.Thread | None = None
        self._sock = None

    WHAT = "the Windows mic helper"
    HINT = "launch go2.bat, which starts it"

    def open(self) -> None:
        with self._cv:
            self._closed = False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="winmic")
            self._thread.start()

    def _run(self) -> None:
        import socket

        while not self._closed:
            try:
                with open(self.token_path, encoding="ascii") as f:
                    token = f.read().strip()
            except OSError:
                self.error = f"{self.WHAT} hasn't started (no {os.path.basename(self.token_path)}): {self.HINT}"
                time.sleep(1.0)
                continue
            sock = socket.socket()
            sock.settimeout(4.0)
            try:
                sock.connect((self.host, self.port))
                sock.sendall(token.encode("ascii"))
            except OSError as e:
                self.error = f"can't reach {self.WHAT} at {self.host}:{self.port} ({type(e).__name__}): {self.HINT}"
                sock.close()
                time.sleep(1.0)
                continue
            self._sock, self.connected, self.error = sock, True, ""
            sock.settimeout(3.0)
            carry = b""
            try:
                while not self._closed:
                    data = sock.recv(6400)
                    if not data:
                        self.error = self.closed_msg()
                        break
                    data = carry + data
                    keep = len(data) - len(data) % 2
                    carry = data[keep:]
                    if keep:
                        self._push(data[:keep])
            except socket.timeout:
                self.error = f"{self.WHAT} stopped sending audio"
            except OSError as e:
                if not self._closed:
                    self.error = f"lost {self.WHAT} ({type(e).__name__})"
            finally:
                self.connected = False
                sock.close()
            if not self._closed:
                time.sleep(1.0)

    def closed_msg(self) -> str:
        return f"the Windows helper closed the connection (it could not open the '{self.device}' microphone? see winmic.log)"

    # push-to-talk (the MicRecorder interface): record from the same capture
    def start(self) -> None:
        self.open()
        with self._cv:
            self._rec = bytearray()

    def stop(self) -> bytes:
        with self._cv:
            rec, self._rec = self._rec, None
        return bytes(rec or b"")

    def close(self) -> None:
        super().close()
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass


class PhoneMic(WinMic):
    """The iPhone's microphone, streamed by phone_server.py (go2.bat starts it). Same socket protocol as WinMic, its own port and secret.
    The phone only sends audio while its page is open and started, so `frames` stays 0 until then and the app keeps its other mic meanwhile."""
    WHAT = "the phone link (phone_server.py)"
    HINT = "launch go2.bat with PHONE=1 set, which starts it"

    def __init__(self, host: str | None = None, port: int = 48125, token_path: str | None = None):
        super().__init__("phone", host, port, token_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".phone_wsl_token"))

    def closed_msg(self) -> str:
        return "the phone link closed the connection (phone_server.py exited?)"


class SwitchMic:
    """The always-on ear and push-to-talk, on the PHONE's microphone whenever the phone is streaming and on the computer's microphone
    otherwise (before the phone page is opened, after its screen locks, when its Wi-Fi drops). It switches by itself, both ways, so the
    wake word keeps working whatever the phone does. `source` says which one is in use right now."""

    LIVE_WITHIN = 1.5                       # s: the phone counts as streaming if audio arrived this recently
    MAX_LAG = 0.25                          # s: never let the phone's audio lag behind real time by more than this

    def __init__(self, phone: "PhoneMic", ear=None, ptt=None):
        self.phone = phone
        self._ear = ear if ear is not None else MicStream()
        self._ptt = ptt if ptt is not None else MicRecorder()
        self._ptt_on = None                 # which one push-to-talk is recording from
        self._was_live = False

    @property
    def live(self) -> bool:
        return self.phone.last_frame_at > 0 and time.time() - self.phone.last_frame_at < self.LIVE_WITHIN

    @property
    def source(self) -> str:
        return "phone" if self.live else "computer"

    def open(self) -> None:
        self.phone.open()
        self._ear.open()

    def _flush_ear(self, ms: int) -> None:
        """The computer's mic buffered sound while the phone was in charge: read that old sound away (fast) so it isn't heard as new."""
        for _ in range(60):
            t0 = time.time()
            if self._ear.read_chunk(ms) is None or time.time() - t0 > 0.5 * ms / 1000:
                return                      # a read took real time: the buffer is empty, we are at 'now'

    def read_chunk(self, ms: int = 100) -> bytes | None:
        if self.live:                       # the phone sets the pace, and its audio is kept close to real time (a backlog is what makes it feel slow)
            self._was_live = True
            with self.phone._cv:
                keep = int(RATE * self.MAX_LAG) * 2
                if len(self.phone._buf) > keep + RATE * ms // 1000 * 2:
                    del self.phone._buf[: len(self.phone._buf) - keep]
            return self.phone.read_chunk(ms)
        if self._was_live:                  # the phone went quiet: forget what it left, and skip the laptop mic's stale sound
            with self.phone._cv:
                self.phone._buf.clear()
            self._was_live = False
            self._flush_ear(ms)
        return self._ear.read_chunk(ms)

    def start(self) -> None:
        self.phone.open()                   # (connects if the ear hasn't already)
        self._ptt_on = self.phone if self.live else self._ptt
        self._ptt_on.start()

    def stop(self) -> bytes:
        src, self._ptt_on = self._ptt_on, None
        return src.stop() if src is not None else b""

    def close(self) -> None:
        self._ear.close()
        self.phone.close()


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def too_short(pcm: bytes) -> bool:
    return len(pcm) < MIN_SECONDS * RATE * 2


# ---- ElevenLabs speech-to-text ----------------------------------------------------------------------------
class ElevenLabsSTT:
    def __init__(self, api_key: str, url: str = STT_URL, model: str = STT_MODEL, language: str | None = STT_LANG):
        self.api_key, self.url, self.model, self.language = api_key, url, model, language

    def _post(self, wav: bytes, model: str):
        import requests

        data = {"model_id": model, "tag_audio_events": "false"}
        if self.language:
            data["language_code"] = self.language
        return requests.post(self.url, headers={"xi-api-key": self.api_key},
                             files={"file": ("speech.wav", wav, "audio/wav")}, data=data, timeout=20)

    def transcribe(self, pcm: bytes) -> str:
        import requests

        wav = wav_bytes(pcm)
        try:
            r = self._post(wav, self.model)
            if r.status_code == 422 and self.model != "scribe_v1":  # model name not accepted: try the older one once
                r = self._post(wav, "scribe_v1")
        except requests.exceptions.ConnectionError as e:
            raise VoiceError("can't reach ElevenLabs: no internet? (the dog's Wi-Fi has none)") from e
        except requests.exceptions.Timeout as e:
            raise VoiceError("ElevenLabs timed out") from e
        if r.status_code in (401, 403):
            raise VoiceError("ElevenLabs rejected the API key (check ELEVENLABS_API_KEY / speech-to-text permission)")
        if r.status_code == 429:
            raise VoiceError("ElevenLabs rate limit / quota reached")
        if r.status_code >= 400:
            raise VoiceError(f"ElevenLabs error {r.status_code}: {r.text[:120]}")
        try:
            j = r.json()
        except ValueError as e:
            raise VoiceError("ElevenLabs sent back something that isn't JSON") from e
        if "transcripts" in j:                          # multichannel shape
            return " ".join(t.get("text", "") for t in j["transcripts"]).strip()
        return (j.get("text") or "").strip()


# ---- local Whisper speech-to-text (offline) ---------------------------------------------------------------
WHISPER_MODEL = os.environ.get("GO2_WHISPER_MODEL", "small.en")   # noisy-speech test: 71/108 vs base.en 61/108, ~2.4x slower. base.en is 32/32 on clean speech


def whisper_model_path(name: str = WHISPER_MODEL) -> str | None:
    """Folder of the cached model, or None if it hasn't been downloaded (never touches the network)."""
    try:
        from faster_whisper.utils import download_model

        return download_model(name, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached / faster-whisper missing
        return None


def fetch_whisper_model(name: str = WHISPER_MODEL) -> str:
    """Download the model into the Hugging Face cache. Needs internet, once."""
    from faster_whisper.utils import download_model

    print(f"Downloading the Whisper '{name}' speech model ...", flush=True)
    path = download_model(name)
    print(f"done: {path}", flush=True)
    return path


class LocalWhisperSTT:
    """faster-whisper on CPU. Loads lazily (or call load() early); transcribe() never touches the network.

    Tuned for short push-to-talk commands: greedy decoding, a single temperature (no fallback re-decodes, which
    can take 20+ s on noise), no vocabulary prompt (measured to make noise slower, not the results better).
    """

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()   # the always-on ear and hold-V push-to-talk share this one model

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            path = whisper_model_path(self.model_name)
            if path is None:
                raise VoiceError("Voice model not downloaded. While ONLINE run: go2.bat --fetch-model")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise VoiceError(f"faster-whisper isn't installed: {e}") from e
            self._model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=max(1, min(8, os.cpu_count() or 4)))

    def transcribe(self, pcm: bytes) -> str:
        import numpy as np

        self.load()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        with self._infer_lock:
            segments, _ = self._model.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                                 condition_on_previous_text=False, vad_filter=False,
                                                 without_timestamps=True,   # one decode pass, no multi-chunk re-tries
                                                 max_new_tokens=48)         # commands are short: bounds time and runaway output
            # Whisper's own "this is noise, not speech" test: a segment it gave a high no-speech probability AND a poor
            # confidence is a hallucination ("thank you.", "you", a repeated word), not something the person said.
            text = " ".join(s.text for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)).strip()
        return sane_transcript(text, len(audio) / RATE)


class SwitchSTT:
    """Two Whisper models: a FAST one (base.en, ~0.5-1 s here) while the phone's close microphone is in use, where the sound is clean, and the
    more accurate slower one (small.en, ~2-3 s here) otherwise (the laptop mic in a noisy room). The speech-to-text is most of the delay after
    you speak, and Whisper always processes a fixed 30 s window whatever the clip length, so only a smaller model is faster."""

    def __init__(self, fast, accurate, use_fast):
        self.fast, self.accurate, self.use_fast = fast, accurate, use_fast
        self.last_used = ""

    def load(self) -> None:
        self.accurate.load()
        try:
            self.fast.load()
        except Exception:  # noqa: BLE001 - the fast one is optional
            self.fast = None

    def transcribe(self, pcm: bytes) -> str:
        m = self.fast if (self.fast is not None and self.use_fast()) else self.accurate
        self.last_used = m.model_name
        return m.transcribe(pcm)


def sane_transcript(text: str, seconds: float) -> str:
    """Drop Whisper decoding loops ('a little bit of a little bit of ...'): far more words than anyone can say in
    that time, or heavily repetitive text. Real commands are a handful of words."""
    import zlib

    if not text:
        return ""
    if len(text.split()) > max(8, 6.0 * seconds):
        return ""
    raw = text.encode()
    if len(raw) > 40 and len(raw) / len(zlib.compress(raw)) > 2.4:
        return ""
    return text


# ---- phrase matcher ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Intent:
    kind: str          # "stop" | "stop_follow" | "follow" | "heel" | "lead" | "sport" | "routine" | "move" | "upright" | "box_step"
    arg: str = ""      # sport command / routine name; move: forward|back|strafe_*|turn_*; upright: on|off; lead: "AHEAD,LEFT" metres, "" = the usual spot
    label: str = ""    # for the on-screen message
    amount: float | None = None   # move only: the number that was spoken, if any
    unit: str = ""                # move only: "s" | "m" | "deg" (as spoken; "" = none given)
    scale: float = 1.0            # move only: "a little" = 0.5, "a lot" = 2.0


_MOTION = object()  # marks where the movement rules sit in the priority order (see _parse_motion)
_LEAD = object()    # ...and where "lead me ..." sits (see _parse_lead): before 'follow' and before the movement rules, which would take "lead me forward 3 metres" as a walk

# Standing on the back legs (Unitree "WalkUpright"). OFF is checked before ON so "stop standing on two legs" comes down.
# Both sit ahead of the 'stop', 'back' (move) and 'stand' (stand up) rules, which would otherwise swallow them.
_LEGS = r"(?:back|hind|rear)\s+(?:legs?|feet|paws)|(?:two|2)\s+(?:legs?|feet|paws)|hind\s*legs?"
_UP_OFF = (r"\b(?:four|4)\s+(?:legs?|paws|feet)\b|\ball\s+fours?\b|\bcome down\b|\bstand normal(?:ly)?\b|\bnormal stance\b"
           r"|\b(?:stop|quit|end|no more)\s+(?:standing|stand|walking|walk|being)\b.*\b(?:" + _LEGS + r"|upright)\b"
           r"|\b(?:down|off)\s+(?:from\s+)?(?:your\s+)?(?:" + _LEGS + r")\b|\bback (?:on|to) (?:all )?four\b")
_UP_ON = (r"\b(?:stand|standing|walk|walking|balance|rear|rise|get|go|be|up)\b.*\b(?:" + _LEGS + r"|upright)\b"
          r"|\brear up\b|\bstand upright\b|\bwalk upright\b|\bbipedal\b|\bon (?:your )?(?:" + _LEGS + r")\b")

_HEEL = (r"(?:\bheel\w*|\bheal\w*|\bhe ll\b|\b(?:walk|stay|come|be)\s+(?:with|beside|next to|by)\s+me\b"
         r"|\bat my side\b|\bby my side\b|\bside by side\b)")

# Ordered: the first matching rule wins, so the safety words come first.
_RULES: list = [
    (r"\b(stop|halt|freeze|emergency|abort)\b.*\b(follow|heel|heal|lead\w*|guid\w*)|\b(follow|heel|heal)\w*\b.*\b(stop|halt|off)\b"
     r"|\b(don t|do not|quit|cancel) (follow|heel|heal|lead\w*|guid\w*)",
     Intent("stop_follow", label="stop following")),
    # "box step" = up on the back legs and step in a square (Whisper also writes it as box stop/stap/tap, so those count too).
    # Ahead of the 'stop' rule.
    (r"\bbox\s*-?\s*(?:step|steps|stap|stab|stop|tap)\b|\bboxstep\b", Intent("box_step", label="box step")),
    (_UP_OFF, Intent("upright", "off", "come down to four legs")),
    (_UP_ON, Intent("upright", "on", "stand on the back legs")),
    (r"\b(stop|halt|freeze|emergency|abort|whoa)\b", Intent("stop", label="STOP")),
    # "ready to dance" = the voice command for standing up (alias; "stand up" still works). Before the 'dance' rules.
    (r"\bready (?:to|for) (?:the )?danc\w*\b|\bready to boogie\b", Intent("sport", "StandUp", "Stand up (ready to dance)")),
    # "heel" = walk at my side. Whisper writes it as heal / he'll (-> "he ll"); "heel right" puts the dog on the right.
    (_HEEL + r".*\bright\b", Intent("heel", "right", "heel (dog on your right)")),
    (_HEEL + r".*\bleft\b", Intent("heel", "left", "heel (dog on your left)")),
    (_HEEL, Intent("heel", "", "heel: walk at your side")),
    (_LEAD, None),
    (r"\bfollow\w*\b", Intent("follow", label="follow the nearest person")),
    (r"\brecover\w*\b|\bget back up\b", Intent("sport", "RecoveryStand", "Recovery stand")),  # before 'back' = move back
    (_MOTION, None),
    (r"\bdanc\w*\s+(2|two|to|too|second)\b|\bsecond dance\b|\bother dance\b", Intent("sport", "Dance2", "Dance 2")),
    (r"\bdanc\w*\b|\bboogie\b|\bgroove\b|\bshow (me )?(your )?moves\b", Intent("sport", "Dance1", "Dance 1")),
    (r"\bgreeting\b|\bintroduce yourself\b", Intent("routine", "greeting", "greeting routine")),
    (r"\bwiggle\b|\bshake (your )?(hips|butt|booty)\b|\bwag\b", Intent("sport", "WiggleHips", "Wiggle hips")),
    (r"\bheart\b|\bi love you\b", Intent("sport", "FingerHeart", "Finger heart")),
    (r"\bstretch\b", Intent("sport", "Stretch", "Stretch")),
    (r"\bhello\b|\bhi\b|\bhey\b|\bwave\b|\bsay hi\b|\bgreet\b", Intent("sport", "Hello", "Hello")),
    (r"\bhappy\b|\bcontent\b|\bgood (boy|dog|girl)\b", Intent("sport", "Content", "Content")),
    (r"\brise\b|\bget up from (sitting|sit)\b", Intent("sport", "RiseSit", "Rise from sit")),
    (r"\bsit\b", Intent("sport", "Sit", "Sit")),
    (r"\b(lie|lay|lying) down\b|\bget down\b|\bdown\b", Intent("sport", "StandDown", "Lie down")),
    (r"\bstand\b|\bget up\b|\bon your feet\b", Intent("sport", "StandUp", "Stand up")),
    (r"\bbalance\b|\bready\b", Intent("sport", "BalanceStand", "Balance stand")),
]


_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
          "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _words_to_digits(t: str) -> str:
    """'turn left forty five degrees' -> 'turn left 45 degrees' (runs of number words become one number)."""
    out, run = [], []

    def flush():
        if run:
            cur = 0
            for w in run:
                if w == "hundred":
                    cur = (cur or 1) * 100
                else:
                    cur += _UNITS.get(w, 0) + _TENS.get(w, 0)
            out.append(str(cur))
            run.clear()

    for w in t.split():
        if w in _UNITS or w in _TENS or (w == "hundred" and run):
            run.append(w)
        else:
            flush()
            out.append(w)
    flush()
    return " ".join(out)


def _parse_motion(t: str) -> Intent | None:
    """Walk/turn phrases: 'walk forward', 'go back a little', 'turn left', 'turn right 45 degrees', 'step left'."""
    t = _words_to_digits(t)
    side = re.search(r"\b(left|right)\b", t)
    side = side.group(1) if side else None
    if re.search(r"\b(turn|rotate|spin|pivot|face)\b", t):
        if re.search(r"\b(around|about|180)\b", t):
            arg, label = "turn_around", "turn around"
        elif side:
            arg, label = f"turn_{side}", f"turn {side}"
        else:
            return None
    elif side and re.search(r"\b(strafe|straf\w*|strap|struff|strife|sidestep|side step|slide|shuffle|step|go|walk|move|head|scoot|shift)\b", t):
        arg, label = f"strafe_{side}", f"step {side}"
    elif re.search(r"\b(back|backward|backwards|reverse|retreat)\b", t):
        arg, label = "back", "walk backward"
    elif re.search(r"\b(forward|forwards|straight|advance)\b|\b(?:walk|move|step|head|run|march) ahead\b", t):   # not "go ahead" = proceed
        arg, label = "forward", "walk forward"
    else:
        return None
    m = re.search(r"(\d+(?:_\d+)?)\s*(seconds?|secs?|meters?|metres?|degrees?|deg)?\b", t)
    amount, unit = None, ""
    if m:
        amount = float(m.group(1).replace("_", "."))
        u = (m.group(2) or "")
        unit = "s" if u.startswith("sec") else "m" if u.startswith(("meter", "metre")) else "deg" if u.startswith("deg") else ""
    scale = 0.5 if re.search(r"\b(little|bit|slightly|tiny|small|short)\b", t) else 2.0 if re.search(r"\b(lot|far|long|way)\b", t) else 1.0
    return Intent("move", arg, label, amount=amount, unit=unit, scale=scale)


MAX_LEAD_METERS = 15.0    # a spoken lead never goes further than this in either direction

_LEAD_WORDS = r"\b(?:lead|leads|leed|led|leading|guide|guides|guiding)\b"     # (Whisper hears "lead" as "led" / "leed")
_LEAD_DIRS = {"forward": "f", "forwards": "f", "ahead": "f", "straight": "f", "left": "l", "right": "r"}


def _parse_lead(t: str) -> Intent | None:
    """'lead' / 'lead me' / 'lead me to the door': the usual spot (arg ""). 'lead me five metres', 'lead me forward 4 metres and left 2',
    'guide me three metres to the right', 'lead me ten feet': arg "AHEAD,LEFT" in metres (left is positive). Each number goes with the
    nearest direction word (ties: the one before it); a number with no direction word is forward. Seconds and degrees are not distances."""
    if not re.search(_LEAD_WORDS, t):
        return None
    toks = _words_to_digits(t).split()
    dirs = [(i, _LEAD_DIRS[w]) for i, w in enumerate(toks) if w in _LEAD_DIRS]
    ahead = left = 0.0
    found = False
    for i, w in enumerate(toks):
        if not re.fullmatch(r"\d+(?:_\d+)?", w):
            continue
        val = float(w.replace("_", "."))
        unit = toks[i + 1] if i + 1 < len(toks) else ""
        if unit.startswith(("sec", "deg", "min", "hour", "percent")):
            continue
        if unit in ("feet", "foot", "ft"):
            val *= 0.3048
        near = min(dirs, key=lambda d: (abs(d[0] - i), d[0] > i), default=None)
        kind = near[1] if near else "f"
        val = min(val, MAX_LEAD_METERS)
        if kind == "f":
            ahead += val
        else:
            left += val if kind == "l" else -val
        found = True
    if not found or (abs(ahead) < 1e-6 and abs(left) < 1e-6):
        return Intent("lead", "", "lead: walk to the usual spot")
    ahead, left = max(-MAX_LEAD_METERS, min(ahead, MAX_LEAD_METERS)), max(-MAX_LEAD_METERS, min(left, MAX_LEAD_METERS))
    side = "" if abs(left) < 1e-6 else f", {abs(left):.1f} m {'left' if left > 0 else 'right'}"
    return Intent("lead", f"{ahead:.2f},{left:.2f}", f"lead: {ahead:.1f} m ahead{side}")


def plan_motion(intent: Intent, linear: float, angular: float) -> tuple[tuple[float, float, float], float]:
    """(vx, vy, yaw), seconds for a 'move' intent, at the given base speeds. Durations are capped for safety."""
    import math

    vec = {"forward": (linear, 0.0, 0.0), "back": (-linear, 0.0, 0.0),
           "strafe_left": (0.0, linear, 0.0), "strafe_right": (0.0, -linear, 0.0),
           "turn_left": (0.0, 0.0, angular), "turn_right": (0.0, 0.0, -angular), "turn_around": (0.0, 0.0, angular)}[intent.arg]
    if intent.arg.startswith("turn"):
        default_deg = 180.0 if intent.arg == "turn_around" else 90.0
        deg = intent.amount if intent.amount is not None and intent.unit in ("", "deg") else default_deg * intent.scale
        secs = math.radians(abs(deg)) / max(angular, 1e-3)
        cap = MAX_TURN_SECONDS
    else:
        default_s = 1.0 if intent.arg.startswith("strafe") else 1.5
        if intent.amount is not None and intent.unit == "m":
            secs = intent.amount / max(linear, 1e-3)
        elif intent.amount is not None and intent.unit in ("", "s"):
            secs = intent.amount
        else:
            secs = default_s * intent.scale
        cap = MAX_WALK_SECONDS
    return vec, min(max(secs, 0.3), cap)


def turn_target(intent: Intent) -> float | None:
    """The angle a spoken turn asks for, in radians (+ = left / counter-clockwise), or None when it is not an angle (a turn given in seconds,
    or not a turn). "turn left" = 90 degrees left, "turn right" = 90 right, "turn around" = 180, "turn right 45 degrees" = 45; "a little" halves it, "a lot" doubles it."""
    import math

    if intent.kind != "move" or not intent.arg.startswith("turn") or intent.unit not in ("", "deg"):
        return None
    default = 180.0 if intent.arg == "turn_around" else 90.0
    deg = abs(intent.amount) if intent.amount is not None else default * intent.scale
    deg = min(deg, math.degrees(MAX_TURN_SECONDS * 0.8))                  # never more than a bit over one full turn
    return math.radians(deg) * (-1.0 if intent.arg == "turn_right" else 1.0)


class TurnController:
    """Closed-loop turning on the spot, by the dog's own gyro. Spins at max_rate, eases off as the angle nears (stopping a little early, by how fast it
    is really turning, because the dog keeps coasting), waits for it to settle, and if it landed more than ~4 degrees off it creeps the last bit.
    step(turned, elapsed) -> (yaw command rad/s, state), state 'run' | 'done' | 'wrong_way' | 'not_moving' | 'timeout'.
    'wrong_way' / 'not_moving' mean the gyro disagrees with what the dog is doing (the caller then falls back to turning by time)."""

    WINDOW = 0.2            # s over which the turning speed is measured
    FIX_RATE = 0.25         # rad/s for the last creep

    def __init__(self, target: float, max_rate: float, min_rate: float = 0.3, gain: float = 2.4, tol_deg: float = 3.0, lag: float = 0.4, max_fixes: int = 2, fix_deg: float = 7.0):
        self.target, self.max_rate, self.min_rate, self.gain, self.lag, self.max_fixes = target, max_rate, min_rate, gain, lag, max_fixes
        self.tol = tol_deg * 3.14159265 / 180.0
        self.fix = fix_deg * 3.14159265 / 180.0                            # landing further off than this gets a creep to correct it
        self.fixes, self.phase, self.settle_t0, self.calm_t0 = 0, "run", 0.0, None
        self.hist: list = []
        self.timeout = 1.6 * abs(target) / max(max_rate, 1e-3) + 4.0

    def _speed(self, elapsed: float, turned: float) -> float:
        self.hist.append((elapsed, turned))
        self.hist = [h for h in self.hist if h[0] >= elapsed - 0.5]
        old = [h for h in self.hist if h[0] <= elapsed - self.WINDOW * 0.75]
        if not old:
            return 0.0
        t0, y0 = old[-1]
        return (turned - y0) / max(elapsed - t0, 1e-3)

    def step(self, turned: float, elapsed: float) -> tuple[float, str]:
        sgn = 1.0 if self.target >= 0 else -1.0
        speed = self._speed(elapsed, turned)                              # signed rad/s, measured
        if self.fixes == 0 and self.phase == "run":
            if elapsed > 1.0 and turned * sgn < -0.17:                    # 10 degrees the WRONG way
                return 0.0, "wrong_way"
            if elapsed > 1.5 and abs(turned) < 0.05:                      # 3 degrees in 1.5 s: it is not turning (or the gyro is dead)
                return 0.0, "not_moving"
        if elapsed > self.timeout:
            return 0.0, "timeout"
        err = self.target - turned                                        # signed radians still to go
        if self.phase == "settle":
            if abs(speed) < 0.08:
                self.calm_t0 = elapsed if self.calm_t0 is None else self.calm_t0
            else:
                self.calm_t0 = None
            calm = self.calm_t0 is not None and elapsed - self.calm_t0 >= 0.15
            if calm or elapsed - self.settle_t0 > 1.0:
                if abs(err) <= self.fix or self.fixes >= self.max_fixes:
                    return 0.0, "done"
                self.fixes, self.phase = self.fixes + 1, "fix"            # landed off: creep the rest
            else:
                return 0.0, "run"
        pred = err - speed * (self.lag if self.phase == "run" else 0.15)  # where it will end up if we stop asking now
        if self.phase == "run":
            if pred * sgn <= self.tol:
                self.phase, self.settle_t0, self.calm_t0 = "settle", elapsed, None
                return 0.0, "run"
            return sgn * min(max(self.gain * pred * sgn, self.min_rate), self.max_rate), "run"
        if abs(pred) <= self.tol:                                         # creeping
            self.phase, self.settle_t0, self.calm_t0 = "settle", elapsed, None
            return 0.0, "run"
        return (1.0 if pred > 0 else -1.0) * self.FIX_RATE, "run"


MAX_WALK_SECONDS = 5.0    # a voice walk never lasts longer than this (~2 m at 0.4 m/s)
MAX_TURN_SECONDS = 8.0    # ~ one full turn at 0.8 rad/s


def normalize(text: str) -> str:
    """Lower-case, strip punctuation, keep decimals as one token ('1.5' -> '1_5')."""
    t = re.sub(r"(?<=\d)\.(?=\d)", "_", text.lower().replace("'", " "))
    t = re.sub(r"[^a-z0-9_ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_command(text: str) -> Intent | None:
    t = normalize(text)
    if not t:
        return None
    for pattern, intent in _RULES:
        if pattern is _MOTION:
            found = _parse_motion(t)
            if found:
                return found
        elif pattern is _LEAD:
            found = _parse_lead(t)
            if found:
                return found
        elif re.search(pattern, t):
            return intent
    return None


# ---- wake word + always-listening -------------------------------------------------------------------------
WAKE_WORD = os.environ.get("GO2_WAKE_WORD", "ernest")
# Whisper spells names creatively; these are accepted for the default wake word (extend after real-world testing).
_WAKE_VARIANTS = {"ernest": ("ernest", "earnest", "ernst", "ernes", "urnest", "ernesto", "ernie")}


def _wake_regex(word: str):
    variants = _WAKE_VARIANTS.get(word.lower(), (word.lower(),))
    return re.compile(r"^(?:(?:hey|ok|okay|hi|yo)\s+)?(?:" + "|".join(re.escape(x) for x in variants) + r")\b\s*(.*)$")


def strip_wake(text: str, word: str = WAKE_WORD) -> tuple[bool, str]:
    """(True, rest) when the utterance STARTS with the wake word ('Ernest, sit down' -> 'sit down'); else (False, text)."""
    t = normalize(text)
    m = _wake_regex(word).match(t)
    return (True, m.group(1).strip()) if m else (False, t)


def is_confirm(text: str) -> bool:
    return bool(re.match(r"^(?:yes|yeah|yep|yup|confirm|confirmed|do it|go ahead|affirmative|okay|ok|sure)\b", normalize(text)))


def route_utterance(text: str, word: str = WAKE_WORD, wake_active: bool = False, confirm_pending: bool = False) -> tuple[str, str]:
    """Decide what an always-listening utterance is. Returns (action, command_text):
       'wake'    the wake word alone: wait for the next utterance
       'command' act on command_text (had the wake word, or the wake window is open)
       'stop'    a bare 'stop' / 'stop following' (short, no wake word needed: a false stop is harmless)
       'confirm' a bare 'yes' while something is waiting for confirmation
       'ignore'  ordinary conversation
    """
    woke, rest = strip_wake(text, word)
    if woke:
        if not rest:
            return "wake", ""
        return ("confirm", "") if confirm_pending and is_confirm(rest) and len(rest.split()) <= 3 else ("command", rest)
    t = rest  # (normalised text)
    intent = parse_command(t)
    if intent and intent.kind in ("stop", "stop_follow") and len(t.split()) <= 4:
        return "stop", t
    if wake_active and t:
        return "command", t
    if confirm_pending and is_confirm(t) and len(t.split()) <= 3:
        return "confirm", ""
    return "ignore", t


class UtteranceSegmenter:
    """Cuts a continuous 16 kHz PCM16 stream into utterances using the Silero VAD bundled with faster-whisper.
    Pure logic (timing comes from sample counts, not the clock) so it can be tested with recorded audio."""

    def __init__(self, end_silence: float = 0.7, max_utterance: float = 12.0, preroll: float = 0.3,
                 tail: float = 0.3, check_every: float = 0.3):
        from faster_whisper.vad import VadOptions

        self.opts = VadOptions(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=500, speech_pad_ms=200)
        self.end_silence, self.max_utt, self.preroll, self.tail, self.check_every = end_silence, max_utterance, preroll, tail, check_every
        self._buf = None
        self._since = 0

    def feed(self, pcm: bytes) -> list[bytes]:
        import numpy as np
        from faster_whisper.vad import get_speech_timestamps

        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = a if self._buf is None else np.concatenate([self._buf, a])
        self._since += len(a)
        if self._since < self.check_every * RATE:
            return []
        self._since = 0
        segs = get_speech_timestamps(self._buf, self.opts)
        if not segs:
            self._buf = self._buf[-RATE:]                       # nothing heard: keep only the last second (pre-roll)
            return []
        last_end, first_start = segs[-1]["end"], segs[0]["start"]
        if (len(self._buf) - last_end) / RATE < self.end_silence and len(self._buf) < self.max_utt * RATE:
            return []                                           # still talking
        s = max(0, first_start - int(self.preroll * RATE))
        e = min(len(self._buf), last_end + int(self.tail * RATE))
        utt = (self._buf[s:e] * 32767).astype(np.int16).tobytes()
        self._buf = self._buf[e:]
        return [utt]


class AlwaysListener:
    """Keeps the mic open, cuts utterances, transcribes each one, and hands the text to on_text(text).
    Everything runs locally. Set `paused` (e.g. while push-to-talk is held) to drop utterances."""

    def __init__(self, stt, on_text, on_error=None, mic=None, segmenter: UtteranceSegmenter | None = None,
                 log_dir: str | None = None):
        self.stt, self.on_text, self.on_error = stt, on_text, on_error
        self.log_dir = log_dir                                  # save every utterance (wav + transcript) here, to study misses
        self.mic = mic or MicStream()
        self.seg = segmenter or UtteranceSegmenter()
        self.paused = False
        self.last_latency = 0.0                                 # s from the end of the last utterance to its transcript (queue wait + speech-to-text)
        self.last_stt = 0.0                                     # s of that spent in the speech-to-text itself
        self.stale_after = 6.0                                  # s: an utterance that waited longer than this is dropped, not acted on late
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.mic.open()
        self._stop.clear()
        self._threads = [threading.Thread(target=self._read_loop, daemon=True, name="ear-read"),
                         threading.Thread(target=self._work_loop, daemon=True, name="ear-stt")]
        for t in self._threads:
            t.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            chunk = self.mic.read_chunk()
            if chunk is None:
                if not self._stop.is_set() and self.on_error:
                    self.on_error("microphone stopped delivering audio")
                return
            for utt in self.seg.feed(chunk):
                if not self.paused:
                    item = (utt, time.time())
                    while True:
                        try:
                            self._q.put_nowait(item)
                            break
                        except queue.Full:                      # STT is behind: drop the OLDEST, the newest speech is what matters
                            try:
                                self._q.get_nowait()
                            except queue.Empty:
                                pass

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utt, t_cut = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if time.time() - t_cut > self.stale_after:
                continue                                        # said long ago: acting on it now would surprise you
            try:
                t0 = time.time()
                text = self.stt.transcribe(utt)
                self.last_stt = time.time() - t0
                self.last_latency = time.time() - t_cut
                self._log(utt, text)
                if text:
                    self.on_text(text)
            except Exception as e:  # noqa: BLE001 - keep listening no matter what one utterance did
                if self.on_error:
                    self.on_error(f"transcription failed: {e}")

    def _log(self, utt: bytes, text: str) -> None:
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            stem = os.path.join(self.log_dir, time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}")
            with open(stem + ".wav", "wb") as f:
                f.write(wav_bytes(utt))
            with open(stem + ".txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass                                                # logging must never stop the ear

    def stop(self) -> None:
        self._stop.set()
        self.mic.close()
