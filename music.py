"""Play songs through the dog's own speaker (its "audio hub"), at a fixed low volume.

How it works (Unitree's audio hub, over the same WebRTC data channel as everything else):
  * songs live in the `music/` folder next to this file (any wav/mp3/m4a/ogg/flac; NOT committed to git)
  * a song is shrunk (mono, 22.05 kHz, 16-bit, trimmed to MAX_SECONDS) and uploaded to the dog ONCE in small chunks;
    the dog remembers it, so later plays are instant. `preload()` does the uploads in the background.
  * play / pause / resume use the audio hub; volume uses the VUI service (api 1003).

Untested on a real dog: whether an Air has a working speaker, and the exact volume scale (assumed 0-10 like the
LED brightness, so 40% = level 4). The chunked-upload format follows unitree_webrtc_connect's own uploader.
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import io
import json
import os
import threading
import time
import wave

MUSIC_DIR = os.environ.get("GO2_MUSIC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "music"))
EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
MAX_SECONDS = float(os.environ.get("GO2_MUSIC_MAX_SECONDS", "90"))   # keeps the one-time upload to a couple of minutes
SAMPLE_RATE = 22050
CHUNK = 4096                    # base64 characters per upload block (same as the library)
TOPIC_AUDIO = "rt/api/audiohub/request"
TOPIC_VUI = "rt/api/vui/request"
API_LIST, API_PLAY, API_PAUSE, API_RESUME, API_MODE, API_UPLOAD = 1001, 1002, 1003, 1004, 1007, 2001
API_SET_VOLUME = 1003           # on the VUI service (not the audio hub)
TRIGGER_PHRASES = {"box step"}  # phrases that mean "play a song" even if no track has that name


class MusicError(Exception):
    """A problem with a friendly message, safe to show in the window."""


def list_tracks(folder: str = MUSIC_DIR) -> list[str]:
    try:
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    except OSError:
        return []


def track_name(filename: str) -> str:
    return os.path.splitext(filename)[0]


def remote_name(filename: str) -> str:
    """The name stored on the dog: plain letters/digits/space/-/_ only, so odd characters can't trip up its file list."""
    import re

    name = re.sub(r"[^A-Za-z0-9 _\-]", "", track_name(filename))
    return re.sub(r"\s+", " ", name).strip()[:48] or "song"


def find_track(query: str, folder: str = MUSIC_DIR) -> str | None:
    """Best filename for a spoken query ('thunderstruck', 'the dance track'); None if nothing is close."""
    files = list_tracks(folder)
    q = query.lower().strip()
    if not files or not q:
        return None
    names = {f: track_name(f).lower().replace("_", " ").replace("-", " ") for f in files}
    for f, n in names.items():
        if n == q:
            return f
    for f, n in names.items():
        if q in n or n in q:
            return f
    close = difflib.get_close_matches(q, list(names.values()), n=1, cutoff=0.6)
    return next((f for f, n in names.items() if close and n == close[0]), None)


def prepare_wav(path: str, max_seconds: float = MAX_SECONDS) -> tuple[bytes, float, bool]:
    """Convert any audio file to a small mono 22.05 kHz 16-bit WAV. Returns (wav_bytes, seconds, was_trimmed)."""
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise MusicError(f"pydub isn't installed: {e}") from e
    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:  # noqa: BLE001 - ffmpeg missing / unreadable file
        raise MusicError(f"couldn't read {os.path.basename(path)}: {e}") from e
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    trimmed = len(audio) / 1000 > max_seconds
    if trimmed:
        audio = audio[: int(max_seconds * 1000)].fade_out(1500)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue(), len(audio) / 1000, trimmed


def upload_blocks(name: str, wav: bytes, now_ms: int | None = None) -> list[dict]:
    """The chunk parameters the dog's UPLOAD_AUDIO_FILE call expects (identical fields to the library's uploader)."""
    b64 = base64.b64encode(wav).decode()
    chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    md5 = hashlib.md5(wav).hexdigest()
    ts = int(time.time() * 1000) if now_ms is None else now_ms
    return [{"file_name": name, "file_type": "wav", "file_size": len(wav), "current_block_index": i,
             "total_block_number": len(chunks), "block_content": c, "current_block_size": len(c),
             "file_md5": md5, "create_time": ts} for i, c in enumerate(chunks, 1)]


def parse_audio_list(response) -> dict[str, str]:
    """{name: unique_id} from a GET_AUDIO_LIST reply. Defensive: the reply nests JSON inside JSON."""
    try:
        data = response["data"]["data"] if isinstance(response, dict) and "data" in response else response
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, str):
            data = json.loads(data)
        items = data.get("audio_list", []) if isinstance(data, dict) else data
        if isinstance(items, str):
            items = json.loads(items)
        out = {}
        for it in items or []:
            name = it.get("CUSTOM_NAME") or it.get("custom_name") or it.get("name")
            uid = it.get("UNIQUE_ID") or it.get("unique_id") or it.get("id")
            if name and uid:
                out[str(name)] = str(uid)
        return out
    except Exception:  # noqa: BLE001
        return {}


def reply_ok(response) -> bool:
    """True unless the dog's reply carries a non-zero status code."""
    try:
        return response["data"]["header"]["status"]["code"] == 0
    except Exception:  # noqa: BLE001 - shape unknown: don't claim failure
        return True


class MusicController:
    """`request(topic, api_id, parameter_json, timeout)` sends one request to the dog and returns its reply."""

    def __init__(self, request, say=print, volume_pct: int = 40, folder: str = MUSIC_DIR):
        self.request, self.say, self.folder = request, say, folder
        self.volume_pct = max(0, min(100, int(volume_pct)))
        self._remote: dict[str, str] = {}
        self._lock = threading.Lock()
        self.playing: str | None = None
        self._gen = 0              # bumped by cancel(): a play() that was still uploading when 'stop' came must not start
        self._starting = False     # True while play() is uploading / starting a song

    def cancel(self) -> bool:
        """'stop' pressed: True if music was playing or on its way; a play() still uploading will now not start."""
        active = self.playing is not None or self._starting
        self._gen += 1
        return active

    # -- volume -----------------------------------------------------------------------------------------
    @staticmethod
    def level(pct: int) -> int:
        return max(0, min(10, round(pct / 10)))        # 0-10 scale assumed (see module docstring)

    def set_volume(self, pct: int | None = None) -> str:
        if pct is not None:
            self.volume_pct = max(0, min(100, int(pct)))
        r = self.request(TOPIC_VUI, API_SET_VOLUME, json.dumps({"volume": self.level(self.volume_pct)}), 8)
        if not reply_ok(r):
            raise MusicError("the dog refused the volume change")
        return f"music volume {self.volume_pct}% (level {self.level(self.volume_pct)} of 10)"

    def bump_volume(self, delta: int) -> str:
        return self.set_volume(self.volume_pct + delta)

    # -- library on the dog -----------------------------------------------------------------------------
    def refresh(self) -> dict[str, str]:
        self._remote = parse_audio_list(self.request(TOPIC_AUDIO, API_LIST, json.dumps({}), 10))
        return self._remote

    def ensure_uploaded(self, filename: str) -> str:
        """Upload the file if the dog doesn't have it yet; returns its unique id."""
        name = remote_name(filename)
        with self._lock:
            self.refresh()
            uid = self._find_remote(name)
            if uid is None:
                path = os.path.join(self.folder, filename)
                wav, secs, trimmed = prepare_wav(path)
                blocks = upload_blocks(name, wav)
                self.say(f"uploading '{name}' to the dog: {secs:.0f}s{' (trimmed)' if trimmed else ''}, "
                         f"{len(blocks)} blocks. One time only ...")
                for b in blocks:
                    r = self.request(TOPIC_AUDIO, API_UPLOAD, json.dumps(b, ensure_ascii=True), 20)
                    if not reply_ok(r):
                        raise MusicError(f"the dog rejected block {b['current_block_index']}/{len(blocks)} of '{name}'")
                    time.sleep(0.02)
                self.refresh()
                uid = self._find_remote(name)
                if uid is None:
                    raise MusicError(f"uploaded '{name}' but the dog doesn't list it (does this dog have a speaker/audio hub?)")
                self.say(f"'{name}' is on the dog now")
            return uid

    def _find_remote(self, name: str) -> str | None:
        """Unique id for a song on the dog: exact name, else a close match (the dog may tidy names)."""
        if name in self._remote:
            return self._remote[name]
        low = {k.lower(): v for k, v in self._remote.items()}
        if name.lower() in low:
            return low[name.lower()]
        close = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.85)
        return low[close[0]] if close else None

    def preload(self, limit: int = 3) -> None:
        """Upload up to `limit` songs in the background so the first play is instant."""
        for f in list_tracks(self.folder)[:limit]:
            try:
                self.ensure_uploaded(f)
            except Exception as e:  # noqa: BLE001
                self.say(f"music preload: {e}")
                return

    # -- playback ---------------------------------------------------------------------------------------
    def choose(self, query: str, default_ok: bool = False) -> str:
        files = list_tracks(self.folder)
        if not files:
            raise MusicError(f"no songs yet: put mp3/wav files in {self.folder}")
        f = find_track(query, self.folder) if query else files[0]
        if f is None and default_ok:
            f = files[0]
        if f is None:
            raise MusicError(f"no song called '{query}'. I have: " + ", ".join(track_name(x) for x in files))
        return f

    def play(self, query: str = "", default_ok: bool = False, loop: bool = False) -> str:
        """Play a song. loop=True repeats it until pause() (used by 'box step': music runs until you say stop)."""
        f = self.choose(query, default_ok)
        gen, self._starting = self._gen, True
        try:
            uid = self.ensure_uploaded(f)
            if gen != self._gen:                                   # 'stop' arrived during the upload: keep the file, don't play
                return "stopped before it started playing"
            return self._start(f, uid, loop)
        finally:
            self._starting = False

    def _start(self, f: str, uid: str, loop: bool) -> str:
        self.set_volume()                                          # always start at the configured volume
        mode = "single_cycle" if loop else "no_cycle"
        self.request(TOPIC_AUDIO, API_MODE, json.dumps({"play_mode": mode}), 8)
        r = self.request(TOPIC_AUDIO, API_PLAY, json.dumps({"unique_id": uid}), 10)
        if not reply_ok(r):
            raise MusicError("the dog refused to play it")
        self.playing = self.last = track_name(f)
        return f"playing '{self.playing}' at {self.volume_pct}% volume" + (" (looping)" if loop else "")

    def pause(self) -> str:
        self.request(TOPIC_AUDIO, API_PAUSE, json.dumps({}), 8)
        self.playing = None
        return "music paused"

    def resume(self) -> str:
        self.request(TOPIC_AUDIO, API_RESUME, json.dumps({}), 8)
        self.playing = getattr(self, "last", None)
        return "music resumed"

    def available(self) -> str:
        files = list_tracks(self.folder)
        return ("songs: " + ", ".join(track_name(f) for f in files)) if files else f"no songs yet: put mp3/wav files in {self.folder}"
