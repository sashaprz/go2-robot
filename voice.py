"""Push-to-talk voice commands for go2.py.

Microphone (WSLg PulseAudio, via libpulse-simple: no apt packages needed) -> ElevenLabs speech-to-text ->
phrase matcher -> an Intent that go2.py turns into the same actions the keys trigger.

Audio only leaves the machine while the push-to-talk key is held. The API key lives in ~/.dimos.env as
ELEVENLABS_API_KEY (see set-elevenlabs-key.bat). It is never stored in the repo.
Needs INTERNET: the dog's own Wi-Fi has none, so use a second connection (see README).
"""
from __future__ import annotations

import ctypes
import io
import os
import re
import threading
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
                pa = ctypes.CDLL("libpulse-simple.so.0")
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

    def start(self) -> None:
        pa = self._lib()
        err = ctypes.c_int(0)
        spec = _SampleSpec(3, RATE, 1)  # 3 = PA_SAMPLE_S16LE
        stream = pa.pa_simple_new(None, b"go2-voice", 2, None, b"push-to-talk",  # 2 = PA_STREAM_RECORD
                                  ctypes.byref(spec), None, None, ctypes.byref(err))
        if not stream:
            raise VoiceError("can't open the microphone: " + pa.pa_strerror(err.value).decode()
                             + " (is WSLg running? Windows mic access allowed?)")
        self._stream, self._frames = stream, []
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


# ---- phrase matcher ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Intent:
    kind: str          # "stop" | "stop_follow" | "follow" | "sport" | "routine"
    arg: str = ""      # sport command name / routine name
    label: str = ""    # for the on-screen message


# Ordered: the first matching rule wins, so the safety words come first.
_RULES: list[tuple[str, Intent]] = [
    (r"\b(stop|halt|freeze|emergency|abort)\b.*\bfollow|\bfollow\w*\b.*\b(stop|halt|off)\b|\b(don t|do not|quit|cancel) follow",
     Intent("stop_follow", label="stop following")),
    (r"\b(stop|halt|freeze|emergency|abort|whoa)\b", Intent("stop", label="STOP")),
    (r"\bfollow\w*\b", Intent("follow", label="follow the nearest person")),
    (r"\bdanc\w*\s+(2|two|to|too|second)\b|\bsecond dance\b|\bother dance\b", Intent("sport", "Dance2", "Dance 2")),
    (r"\bdanc\w*\b|\bboogie\b|\bgroove\b|\bshow (me )?(your )?moves\b", Intent("sport", "Dance1", "Dance 1")),
    (r"\bgreeting\b|\bintroduce yourself\b", Intent("routine", "greeting", "greeting routine")),
    (r"\bwiggle\b|\bshake (your )?(hips|butt|booty)\b|\bwag\b", Intent("sport", "WiggleHips", "Wiggle hips")),
    (r"\bheart\b|\bi love you\b", Intent("sport", "FingerHeart", "Finger heart")),
    (r"\bstretch\b", Intent("sport", "Stretch", "Stretch")),
    (r"\bhello\b|\bhi\b|\bhey\b|\bwave\b|\bsay hi\b|\bgreet\b", Intent("sport", "Hello", "Hello")),
    (r"\bhappy\b|\bcontent\b|\bgood (boy|dog|girl)\b", Intent("sport", "Content", "Content")),
    (r"\brecover\w*\b|\bget back up\b", Intent("sport", "RecoveryStand", "Recovery stand")),
    (r"\brise\b|\bget up from (sitting|sit)\b", Intent("sport", "RiseSit", "Rise from sit")),
    (r"\bsit\b", Intent("sport", "Sit", "Sit")),
    (r"\b(lie|lay|lying) down\b|\bget down\b|\bdown\b", Intent("sport", "StandDown", "Lie down")),
    (r"\bstand\b|\bget up\b|\bon your feet\b", Intent("sport", "StandUp", "Stand up")),
    (r"\bbalance\b|\bready\b", Intent("sport", "BalanceStand", "Balance stand")),
]


def parse_command(text: str) -> Intent | None:
    t = re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", " "))
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    for pattern, intent in _RULES:
        if re.search(pattern, t):
            return intent
    return None
