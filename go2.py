#!/usr/bin/env python3
"""Go2 all-in-one: live camera (FPV) + keyboard driving + poses/tricks + person-follow in ONE window.

Run via go2.bat (after joining the dog's Wi-Fi). Needs no internet. The window must have keyboard focus.
Reuses DimOS's own connection class, so the per-device AES key works exactly like dimos-go2.bat.
The dog accepts ONE controller at a time: don't run this alongside dimos-go2.bat / the phone app.

  python go2.py                connect to the real dog
  python go2.py --demo         same window with a FAKE robot and synthetic video (no dog needed)
  python go2.py --fetch-model  download the object detector + offline voice model (one time, needs internet)
  python go2.py --selftest / --selftest-follow   headless scripted tests (used during development)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

_SELFTEST = any(a.startswith("--selftest") for a in sys.argv)
if _SELFTEST:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
elif sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")  # WSLg; same driver DimOS's keyboard window forces
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

try:
    import follow as follow_mod  # person detector + follow controller (follow.py, next to this file)
except Exception as _e:  # noqa: BLE001 - go2.py must still run without it
    follow_mod = None
    _FOLLOW_ERR = str(_e)

try:
    import voice as voice_mod  # push-to-talk mic + ElevenLabs speech-to-text + phrase matcher (voice.py)
except Exception as _ve:  # noqa: BLE001
    voice_mod = None
    _VOICE_ERR = str(_ve)

# ---- tunables ---------------------------------------------------------------------------------------------
LINEAR = 0.4      # m/s forward / sideways
ANGULAR = 0.8     # rad/s turning
BOOST = 1.5       # Shift multiplier
SLOW = 0.5        # Ctrl multiplier
CONTROL_HZ = 20   # velocity command rate (the connection also self-stops 0.2 s after the last command)
CONFIRM_SECS = 4  # how long a dance/routine/follow waits for the confirm key
FOLLOW_STALE = 0.7  # ignore a follow command older than this many seconds (detector stalled): stand still
VOICE_TIMEOUT = 12  # give up on a transcription after this many seconds

# key -> (label, sport command, seconds the dog is busy afterwards; driving is locked out meanwhile)
TRICKS = {
    pygame.K_1: ("Stand up", "StandUp", 3.0),
    pygame.K_2: ("Balance stand", "BalanceStand", 2.0),
    pygame.K_3: ("Lie down", "StandDown", 3.0),
    pygame.K_4: ("Recovery stand", "RecoveryStand", 4.0),
    pygame.K_5: ("Sit", "Sit", 3.0),
    pygame.K_6: ("Rise from sit", "RiseSit", 3.0),
    pygame.K_7: ("Hello", "Hello", 5.0),
    pygame.K_8: ("Stretch", "Stretch", 6.0),
    pygame.K_9: ("Content", "Content", 5.0),
    pygame.K_0: ("Wiggle hips", "WiggleHips", 5.0),
    pygame.K_f: ("Finger heart", "FingerHeart", 6.0),
}
# Long/showy moves need a second keypress (Y) so a stray key can't start a 20 s dance.
CONFIRM_TRICKS = {
    pygame.K_n: ("Dance 1", "Dance1", 20.0),
    pygame.K_m: ("Dance 2", "Dance2", 20.0),
}
ROUTINE_KEY = pygame.K_r
FOLLOW_KEY = pygame.K_t
VOICE_KEY = pygame.K_v   # hold to talk
OBJECTS_KEY = pygame.K_o  # toggle the object-detection overlay
# EDIT FREELY. Timed lists of the sport commands above; each step waits that command's busy time.
ROUTINES = {"greeting": ["StandUp", "BalanceStand", "Hello", "Content", "WiggleHips", "Sit"]}
# Deliberately absent: flips, handstand, bound. They can hurt the robot and most aren't supported on an Air.

WIN_W, WIN_H = 1024, 760
VIDEO_H = 576
BG = (18, 20, 24)
TXT = (225, 228, 232)
DIM = (140, 146, 154)
WARN = (255, 208, 90)
BAD = (255, 110, 110)
GOOD = (120, 220, 150)


def _lookup():
    table = {}
    for label, name, wait in list(TRICKS.values()) + list(CONFIRM_TRICKS.values()):
        table[name] = (label, wait)
    return table


# ---- robot backends ---------------------------------------------------------------------------------------
class Robot:
    """The real dog, via DimOS's UnitreeWebRTCConnection (connects in __init__)."""

    def __init__(self, ip: str, aes_key: str, motion_mode: str | None = None):
        from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

        from dimos.robot.unitree.connection import UnitreeWebRTCConnection

        self._topic, self._cmd = RTC_TOPIC, SPORT_CMD
        self.c = UnitreeWebRTCConnection(ip, aes_128_key=aes_key)
        if motion_mode:
            self.c.set_motion_mode(motion_mode)
        self._subs: list = []

    def sport(self, name: str) -> None:
        coro = self.c.conn.datachannel.pub_sub.publish_request_new(
            self._topic["SPORT_MOD"], {"api_id": self._cmd[name]}
        )
        asyncio.run_coroutine_threadsafe(coro, self.c.loop).result(timeout=8)

    def move(self, vx: float, vy: float, yaw: float) -> None:
        from dimos.msgs.geometry_msgs.Twist import Twist
        from dimos.msgs.geometry_msgs.Vector3 import Vector3

        t = Twist()
        t.linear = Vector3(vx, vy, 0)
        t.angular = Vector3(0, 0, yaw)
        self.c.move(t)

    def stop_move(self) -> None:
        self.c.stop_movement()

    def on_frame(self, cb) -> None:
        self._subs.append(self.c.raw_video_stream().subscribe(lambda f: cb(f.to_ndarray(format="rgb24"))))

    def on_battery(self, cb) -> None:
        def handle(msg):
            try:
                cb(msg["data"]["bms_state"]["soc"])
            except Exception:  # noqa: BLE001 - odd/partial message: just skip it
                pass

        self._subs.append(self.c.lowstate_stream().subscribe(handle))

    def close(self) -> None:
        for s in self._subs:
            try:
                s.dispose()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.c.stop()  # zero velocity, then disconnect
        except Exception:  # noqa: BLE001
            pass


class FakeRobot:
    """No dog: records commands and streams synthetic (or a supplied still) video, for --demo and selftests."""

    def __init__(self, image: np.ndarray | None = None):
        self.log: list[tuple] = []
        self._halt = threading.Event()
        self.image = image

    def _add(self, entry: tuple) -> None:
        if not self.log or self.log[-1] != entry:  # collapse the 20 Hz repeats
            self.log.append(entry)

    def sport(self, name: str) -> None:
        self._add(("sport", name))

    def move(self, vx: float, vy: float, yaw: float) -> None:
        self._add(("move", round(vx, 2), round(vy, 2), round(yaw, 2)))

    def stop_move(self) -> None:
        self._add(("stop_move",))

    def on_frame(self, cb) -> None:
        def gen():
            h, w = 720, 1280
            yy = np.linspace(40, 200, h, dtype=np.uint8)[:, None, None]
            base = np.repeat(np.repeat(yy, w, axis=1), 3, axis=2)
            base[..., 2] = np.clip(base[..., 2].astype(int) + 40, 0, 255).astype(np.uint8)
            t0 = time.time()
            while not self._halt.is_set():
                if self.image is not None:
                    f = self.image
                else:
                    f = base.copy()
                    x = int(((time.time() - t0) * 200) % (w - 200))
                    f[300:420, x : x + 200] = (255, 190, 60)
                cb(f)
                time.sleep(1 / 15)

        threading.Thread(target=gen, daemon=True).start()

    def on_battery(self, cb) -> None:
        cb(87)

    def close(self) -> None:
        self._halt.set()


class FakeMic:
    """Self-test stand-in for voice.MicRecorder: 'records' one second of silence."""

    def start(self) -> None:
        pass

    def stop(self) -> bytes:
        return b"\x01\x00" * 16000


class FakeSTT:
    """Self-test stand-in for voice.ElevenLabsSTT: returns canned transcripts in order."""

    def __init__(self, lines):
        self.lines = list(lines)

    def transcribe(self, pcm: bytes) -> str:
        return self.lines.pop(0)


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Stable, distinct colour per object class (people are always the same warm orange)."""
    import colorsys

    if class_id == 0:
        return (255, 170, 60)
    r, g, b = colorsys.hsv_to_rgb((class_id * 0.618034) % 1.0, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---- the app ----------------------------------------------------------------------------------------------
class App:
    def __init__(self, args, demo_image=None):
        self.args = args
        self.demo_image = demo_image
        self.robot = None
        self.state = "connecting ..."
        self.state_color = WARN
        self.held: set[int] = set()
        self.armed = False          # has BalanceStand been sent since the last pose/trick?
        self.busy_until = 0.0       # driving is locked out until then
        self.pending = None         # (label, action, deadline) awaiting the Y confirm
        self.abort = threading.Event()
        self.stop_evt = threading.Event()
        self.desired = (0.0, 0.0, 0.0)
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.lookup = _lookup()
        self.messages: deque = deque(maxlen=4)
        self.frame_lock = threading.Lock()
        self.latest = None
        self.frame_new = False
        self.frame_seq = 0
        self.frame_times: deque = deque(maxlen=40)
        self.last_frame_at = 0.0
        self.battery = None
        self._fonts: dict = {}
        self._scaled = None
        self._xf = (0, 0, 1.0)      # (x offset, y offset, scale) from frame pixels to window pixels
        self.running = True
        # person-follow
        self.following = False
        self.follower = None
        self.detector = None
        self._detector_loading = False
        self.follow_res = None      # (timestamp, FollowResult, frame_shape)
        self._log_len_after_stop = None  # selftest bookkeeping
        # voice (push-to-talk)
        self.mic = None
        self.stt = None
        self.voice_state = ""       # "" | "listening" | "transcribing"
        self.voice_q: queue.Queue = queue.Queue()
        self._voice_gen = 0                 # bumped per transcription so a timed-out one can be ignored
        self._voice_deadline = 0.0
        self.heard_log: list = []           # (transcript, matched label) - used by the self-test
        self.follow_starts = 0              # how many times follow actually started - self-test
        # object detection overlay
        self.objects_on = False
        self.objects = None                 # (timestamp, [(x1, y1, x2, y2, score, class_id)], frame_shape)
        self._obj_snapshot = None           # self-test bookkeeping

    # -- feedback ---------------------------------------------------------------------------------------
    def say(self, text: str, color=TXT) -> None:
        print(text, flush=True)
        self.messages.append((time.time(), text, color))

    def font(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def text(self, screen, s: str, x: int, y: int, color=TXT, size: int = 24) -> None:
        screen.blit(self.font(size).render(s, True, color), (x, y))

    # -- robot plumbing ---------------------------------------------------------------------------------
    def connect_bg(self) -> None:
        try:
            if (self.args.demo or self.args.selftest or self.args.selftest_follow or self.args.selftest_voice
                    or self.args.selftest_objects):
                r = FakeRobot(self.demo_image)
            else:
                key = os.environ.get("UNITREE_AES_128_KEY")
                if not key:
                    raise RuntimeError("UNITREE_AES_128_KEY isn't set (expected in ~/.dimos.env)")
                r = Robot(self.args.ip, key, self.args.motion_mode)
            r.on_frame(self.on_frame)
            r.on_battery(lambda soc: setattr(self, "battery", soc))
            self.robot = r
            self.state, self.state_color = ("DEMO (fake robot)" if isinstance(r, FakeRobot) else "connected"), GOOD
            self.say("Connected. Keep the area around the dog clear.", GOOD)
        except Exception as e:  # noqa: BLE001
            self.state, self.state_color = f"connection failed: {e}", BAD
            self.say(f"Connection failed: {e}", BAD)
            self.say("Close the phone app / any running dimos-go2 window, check the Wi-Fi, then retry.", WARN)

    def on_frame(self, arr) -> None:
        with self.frame_lock:
            self.latest = arr
            self.frame_new = True
            self.frame_seq += 1
        now = time.time()
        self.last_frame_at = now
        self.frame_times.append(now)

    def control_loop(self) -> None:
        moving = False
        while not self.stop_evt.is_set():
            r, d = self.robot, self.desired
            if r is not None:
                try:
                    if any(d):
                        r.move(*d)
                        moving = True
                    elif moving:
                        r.stop_move()
                        moving = False
                except Exception as e:  # noqa: BLE001
                    self.say(f"move failed: {e}", BAD)
            time.sleep(1 / CONTROL_HZ)

    def send(self, name: str) -> bool:
        if self.robot is None:
            self.say("not connected yet", WARN)
            return False
        try:
            self.robot.sport(name)
            return True
        except Exception as e:  # noqa: BLE001
            self.say(f"{name} failed: {e}", BAD)
            return False

    # -- person following -------------------------------------------------------------------------------
    def follow_available(self) -> bool:
        if follow_mod is None:
            self.say(f"follow.py couldn't be loaded: {_FOLLOW_ERR}", BAD)
            return False
        if not follow_mod.model_present():
            self.say("Object/person detector not downloaded. While ONLINE run: go2.bat --fetch-model", WARN)
            return False
        return True

    def _load_detector(self) -> None:
        try:
            self.detector = follow_mod.ObjectDetector()
        except Exception as e:  # noqa: BLE001
            self.say(f"detector failed to load: {e}", BAD)
            self.objects_on = False
            self.stop_follow("detector unavailable")
        finally:
            self._detector_loading = False

    def ensure_detector(self) -> None:
        if self.detector is None and not self._detector_loading:
            self._detector_loading = True
            threading.Thread(target=self._load_detector, daemon=True).start()

    def toggle_objects(self) -> None:
        if self.objects_on:
            self.objects_on, self.objects = False, None
            self.say("object detection off")
        elif self.follow_available():
            self.objects_on = True
            self.ensure_detector()
            self.say("object detection on (YOLOX-tiny, 80 everyday object types)", GOOD)

    def start_follow(self) -> None:
        cfg = follow_mod.FollowConfig(max_forward=self.args.follow_speed)
        self.follower = follow_mod.Follower(cfg)
        self.follower.reset()
        self.follow_res = None
        self.following = True
        self.follow_starts += 1
        self.say("> following the nearest person (Space / T / any drive key stops)", GOOD)
        self.ensure_detector()

    def stop_follow(self, reason: str) -> None:
        if not self.following:
            return
        self.following = False
        self.follow_res = None
        self.desired = (0.0, 0.0, 0.0)
        self.say(f"follow stopped: {reason}", WARN)

    def vision_loop(self) -> None:
        """One inference per new frame, shared by person-follow and the object overlay."""
        last_seq = -1
        while not self.stop_evt.is_set():
            det = self.detector
            if not (self.following or self.objects_on) or det is None or self.latest is None or self.frame_seq == last_seq:
                time.sleep(0.03)
                continue
            with self.frame_lock:
                frame, last_seq = self.latest, self.frame_seq
            try:
                dets = det.detect_objects(frame)
                if self.objects_on:
                    self.objects = (time.time(), dets, frame.shape)
                res = None
                if self.following:
                    res = self.follower.step(follow_mod.people(dets, frame.shape[0]), frame.shape)
            except Exception as e:  # noqa: BLE001
                self.say(f"detector error: {e}", BAD)
                self.objects_on = False
                self.stop_follow("detector error")
                continue
            if res is not None:
                self.follow_res = (time.time(), res, frame.shape)
                if res.lost:
                    self.stop_follow(res.status)

    # -- voice (push-to-talk) ---------------------------------------------------------------------------
    def voice_ready(self) -> bool:
        if self.mic is not None and self.stt is not None:  # (self-tests inject fakes)
            return True
        if voice_mod is None:
            self.say(f"voice.py couldn't be loaded: {_VOICE_ERR}", BAD)
            return False
        if self.args.stt == "elevenlabs":
            key = os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                self.say("No ELEVENLABS_API_KEY. Run set-elevenlabs-key.bat, then restart go2.bat", WARN)
                return False
            self.mic, self.stt = voice_mod.MicRecorder(), voice_mod.ElevenLabsSTT(key)
            return True
        if voice_mod.whisper_model_path(self.args.whisper_model) is None:
            self.say("Voice model not downloaded. While ONLINE run: go2.bat --fetch-model", WARN)
            return False
        self.mic, self.stt = voice_mod.MicRecorder(), voice_mod.LocalWhisperSTT(self.args.whisper_model)
        return True

    def _prepare_voice(self) -> None:
        """Load the local Whisper model in the background at startup so the first V press is quick."""
        try:
            if voice_mod is not None and self.args.stt == "local" and voice_mod.whisper_model_path(self.args.whisper_model):
                if self.voice_ready():
                    self.stt.load()
                    self.say("voice model ready (offline Whisper)", GOOD)
        except Exception as e:  # noqa: BLE001
            self.say(f"voice model failed to load: {e}", BAD)

    def start_listening(self) -> None:
        if self.voice_state or not self.voice_ready():
            return
        try:
            self.mic.start()
        except Exception as e:  # noqa: BLE001
            self.say(f"voice: {e}", BAD)
            return
        self.voice_state = "listening"

    def stop_listening(self) -> None:
        if self.voice_state != "listening":
            return
        pcm = self.mic.stop()
        if voice_mod is not None and voice_mod.too_short(pcm):
            self.voice_state = ""
            self.say("too short: hold V the whole time you speak", WARN)
            return
        self.voice_state = "transcribing"
        self._voice_gen += 1
        self._voice_deadline = time.time() + VOICE_TIMEOUT
        threading.Thread(target=self._transcribe, args=(pcm, self._voice_gen), daemon=True).start()

    def cancel_listening(self) -> None:
        if self.voice_state == "listening":
            try:
                self.mic.stop()
            except Exception:  # noqa: BLE001
                pass
            self.voice_state = ""
            self.say("voice cancelled (window lost focus)", WARN)

    def _transcribe(self, pcm: bytes, gen: int) -> None:
        try:
            self.voice_q.put(("heard", self.stt.transcribe(pcm), gen))
        except Exception as e:  # noqa: BLE001 - VoiceError or anything else: show it, never crash the window
            self.voice_q.put(("error", str(e), gen))
        finally:
            if gen == self._voice_gen:
                self.voice_state = ""

    def drain_voice(self) -> None:
        if self.voice_state == "transcribing" and time.time() > self._voice_deadline:
            self._voice_gen += 1      # abandon the slow transcription; its late result will be ignored
            self.voice_state = ""
            self.say("voice: transcription timed out, try again", BAD)
        while True:
            try:
                kind, payload, gen = self.voice_q.get_nowait()
            except queue.Empty:
                return
            if gen != self._voice_gen:
                continue              # a result from a transcription that already timed out
            if kind == "error":
                self.say(f"voice: {payload}", BAD)
            else:
                self.handle_heard(payload)

    def handle_heard(self, text: str) -> None:
        """Turn a transcript into the same actions the keys trigger. Voice is push-to-talk, so it is deliberate:
        tricks run directly, but starting to FOLLOW (autonomous walking) still needs the Y key."""
        intent = voice_mod.parse_command(text) if text else None
        self.heard_log.append((text, intent.label if intent else None))
        if not text:
            self.say("heard nothing", WARN)
            return
        if intent is None:
            self.say(f'heard: "{text}"  (no command matched)', WARN)
            return
        self.say(f'heard: "{text}"  ->  {intent.label}', GOOD)
        if intent.kind == "stop":
            self.emergency_stop()
        elif intent.kind == "stop_follow":
            self.stop_follow("voice")
        elif intent.kind == "follow":
            if not self.following and self.follow_available():
                self.pending = ("Follow the nearest person (NO obstacle avoidance)", self.start_follow, time.time() + CONFIRM_SECS)
        elif intent.kind == "sport":
            label, wait = self.lookup[intent.arg]
            self.run_trick(label, intent.arg, wait)
        elif intent.kind == "routine":
            self.run_routine(intent.arg)

    # -- actions ----------------------------------------------------------------------------------------
    def run_trick(self, label: str, name: str, wait: float) -> None:
        self.stop_follow(f"{label} pressed")
        self.say(f"> {label}")
        self.busy_until = time.time() + wait
        self.armed = name == "BalanceStand"  # after any other pose/trick, re-balance before driving
        self.pool.submit(self.send, name)

    def run_routine(self, rname: str) -> None:
        self.abort.clear()
        self.stop_follow("routine started")

        def work():
            for step in ROUTINES[rname]:
                if self.abort.is_set():
                    break
                label, wait = self.lookup[step]
                self.say(f"> routine '{rname}': {label}")
                self.armed = step == "BalanceStand"
                self.busy_until = time.time() + wait
                self.send(step)
                end = time.time() + wait
                while time.time() < end and not self.abort.is_set():
                    time.sleep(0.05)
            self.say(f"routine '{rname}' {'aborted' if self.abort.is_set() else 'finished'}")

        threading.Thread(target=work, daemon=True).start()

    def emergency_stop(self) -> None:
        self.abort.set()
        self.pending = None
        self.busy_until = 0.0
        self.desired = (0.0, 0.0, 0.0)
        self.stop_follow("SPACE")
        self.say("STOP", BAD)
        self.pool.submit(self.send, "StopMove")

    def on_key(self, key: int) -> None:
        now = time.time()
        self.held.add(key)
        if key == pygame.K_ESCAPE:
            self.running = False
            return
        if key == pygame.K_SPACE:
            self.emergency_stop()
            return
        if self.pending:
            label, action, deadline = self.pending
            self.pending = None
            if key == pygame.K_y and now < deadline:
                action()
                return
            self.say(f"cancelled: {label}")
        if key == FOLLOW_KEY:
            if self.following:
                self.stop_follow("T pressed")
            elif self.follow_available():
                self.pending = ("Follow the nearest person (NO obstacle avoidance)", self.start_follow, now + CONFIRM_SECS)
        elif key == VOICE_KEY:
            self.start_listening()
        elif key == OBJECTS_KEY:
            self.toggle_objects()
        elif key in TRICKS:
            self.run_trick(*TRICKS[key])
        elif key in CONFIRM_TRICKS:
            t = CONFIRM_TRICKS[key]
            self.pending = (t[0], lambda t=t: self.run_trick(*t), now + CONFIRM_SECS)
        elif key == ROUTINE_KEY:
            rname = next(iter(ROUTINES))
            self.pending = (f"routine '{rname}'", lambda: self.run_routine(rname), now + CONFIRM_SECS)

    def ensure_ready(self, now: float) -> bool:
        """True when the dog is balanced and not busy. Sends BalanceStand first if needed (wireless-controller
        velocity only works in BalanceStand)."""
        if now < self.busy_until:
            return False
        if not self.armed:
            self.armed = True
            self.busy_until = now + 1.0
            self.say("> balancing before driving ...")
            self.pool.submit(self.send, "BalanceStand")
            return False
        return True

    def update_velocity(self) -> None:
        now, h = time.time(), self.held
        fwd = (pygame.K_w in h) - (pygame.K_s in h)
        side = (pygame.K_q in h) - (pygame.K_e in h)      # Q = left, E = right
        turn = (pygame.K_a in h) - (pygame.K_d in h)      # A = turn left, D = turn right
        manual = bool(fwd or side or turn)
        if self.robot is None:
            self.desired = (0.0, 0.0, 0.0)
            return
        if self.following:
            if manual:
                self.stop_follow("manual drive key")      # the human always wins; carry on as manual below
            else:
                res = self.follow_res
                if not self.ensure_ready(now) or res is None or now - res[0] > FOLLOW_STALE:
                    self.desired = (0.0, 0.0, 0.0)        # not ready / no fresh detection: stand still
                else:
                    self.desired = res[1].cmd
                return
        if not manual or not self.ensure_ready(now):
            self.desired = (0.0, 0.0, 0.0)
            return
        scale = 1.0
        if h & {pygame.K_LSHIFT, pygame.K_RSHIFT}:
            scale = BOOST
        elif h & {pygame.K_LCTRL, pygame.K_RCTRL}:
            scale = SLOW
        lin, ang = self.args.linear * scale, self.args.angular * scale
        self.desired = (fwd * lin, side * lin, turn * ang)

    # -- drawing ----------------------------------------------------------------------------------------
    def draw(self, screen) -> None:
        screen.fill(BG)
        with self.frame_lock:
            frame, fresh = self.latest, self.frame_new
            self.frame_new = False
        if frame is not None and fresh:
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
            s = min(WIN_W / surf.get_width(), VIDEO_H / surf.get_height())
            self._scaled = pygame.transform.smoothscale(surf, (int(surf.get_width() * s), int(surf.get_height() * s)))
            self._xf = ((WIN_W - self._scaled.get_width()) // 2, (VIDEO_H - self._scaled.get_height()) // 2, s)
        if self._scaled is not None:
            screen.blit(self._scaled, self._xf[:2])
        else:
            self.text(screen, "waiting for video ...", WIN_W // 2 - 110, VIDEO_H // 2, DIM, 32)
        if self.last_frame_at and time.time() - self.last_frame_at > 3:
            self.text(screen, "NO VIDEO", WIN_W // 2 - 70, VIDEO_H // 2 - 20, BAD, 44)

        now = time.time()
        obj = self.objects
        if self.objects_on and obj and now - obj[0] < 1.0:  # labelled boxes for everything the detector sees
            ox, oy, sc = self._xf
            for x1, y1, x2, y2, score, cid in obj[1]:
                color = _class_color(cid)
                pygame.draw.rect(screen, color, (ox + x1 * sc, oy + y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc), 2)
                label = self.font(20).render(f"{follow_mod.COCO_CLASSES[cid]} {score:.0%}", True, (10, 10, 10))
                lw, lh = label.get_size()
                tx, ty = ox + x1 * sc, max(36, oy + y1 * sc - lh)
                pygame.draw.rect(screen, color, (tx, ty, lw + 6, lh))
                screen.blit(label, (tx + 3, ty))
            summary = follow_mod.summarize(obj[1]) or "nothing recognised"
            surf = self.font(24).render("seeing: " + summary, True, TXT)
            back = pygame.Surface((surf.get_width() + 16, 30), pygame.SRCALPHA)
            back.fill((0, 0, 0, 165))
            screen.blit(back, (WIN_W - back.get_width() - 6, VIDEO_H - 36))
            screen.blit(surf, (WIN_W - surf.get_width() - 14, VIDEO_H - 32))
        elif self.objects_on:
            self.text(screen, "object detection: " + ("loading detector ..." if self.detector is None else "no frames yet"),
                      WIN_W - 330, VIDEO_H - 32, WARN, 24)
        res = self.follow_res
        if self.following and res and res[1].box and now - res[0] < 1.0:  # box around the tracked person
            ox, oy, sc = self._xf
            x1, y1, x2, y2 = res[1].box[:4]
            pygame.draw.rect(screen, GOOD if res[1].status != "close enough" else WARN,
                             (ox + x1 * sc, oy + y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc), 3)

        fps = 0.0
        if len(self.frame_times) > 2 and now - self.frame_times[-1] < 2:
            fps = (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-3)
        bar = pygame.Surface((WIN_W, 34), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 150))
        screen.blit(bar, (0, 0))
        bat = f"{self.battery}%" if self.battery is not None else "?"
        self.text(screen, self.state, 12, 8, self.state_color, 24)
        self.text(screen, f"video {fps:.0f} fps    battery {bat}", 360, 8, TXT, 24)
        d = self.desired
        status = "BUSY" if now < self.busy_until else ("balancing" if self.armed else "will balance on first drive key")
        self.text(screen, f"{status}    vx {d[0]:+.2f}  vy {d[1]:+.2f}  yaw {d[2]:+.2f}", 640, 8, TXT if any(d) else DIM, 24)

        banner = None
        if self.voice_state == "listening":
            banner = ("LISTENING ...  release V to send", BAD)
        elif self.voice_state == "transcribing":
            banner = ("transcribing ...", WARN)
        elif self.pending and now < self.pending[2]:
            banner = (f"{self.pending[0]}: press Y to confirm ({self.pending[2] - now:.0f}s)", WARN)
        elif self.following:
            st = res[1].status if res else ("loading detector ..." if self.detector is None else "looking for a person ...")
            banner = (f"FOLLOWING: {st}    (Space / T / any drive key stops)", GOOD)
        if banner:
            back = pygame.Surface((WIN_W, 40), pygame.SRCALPHA)
            back.fill((0, 0, 0, 150))
            screen.blit(back, (0, 36))
            self.text(screen, banner[0], 12, 44, banner[1], 30)

        recent = [(msg, color) for t, msg, color in self.messages if now - t < 8]
        if recent:
            y = VIDEO_H - 26 * len(recent) - 10
            backing = pygame.Surface((620, 26 * len(recent) + 8), pygame.SRCALPHA)
            backing.fill((0, 0, 0, 165))  # readable even over a bright camera frame
            screen.blit(backing, (6, y - 4))
            for msg, color in recent:
                self.text(screen, msg, 12, y, color, 24)
                y += 26

        pygame.draw.rect(screen, (28, 31, 37), (0, VIDEO_H, WIN_W, WIN_H - VIDEO_H))
        lines = [
            "DRIVE   W/S forward/back    Q/E strafe left/right    A/D turn left/right    Shift fast   Ctrl slow   SPACE = STOP",
            "POSES   1 stand up   2 balance   3 lie down   4 recovery stand   5 sit   6 rise from sit",
            "TRICKS  7 hello   8 stretch   9 content   0 wiggle hips   F finger heart   N / M dance 1 / 2 (then Y)   R greeting routine (then Y)",
            "FOLLOW  T follow the nearest person (then Y).  T again / Space / any drive key stops it.  No obstacle avoidance!",
            "VOICE   hold V and talk: \"say hello\", \"dance two\", \"sit\", \"stretch\", \"follow me\" (then Y), \"stop\".  "
            + ("Offline (local Whisper)." if self.args.stt == "local" else "Needs internet (ElevenLabs)."),
            "VISION  O toggles labelled boxes for 80 everyday object types (person, chair, cup, ball, tv, ...).  Small model: misses far/small things.",
            "Click this window so it has keyboard focus.   Esc quits.",
        ]
        for i, line in enumerate(lines):
            self.text(screen, line, 12, VIDEO_H + 10 + i * 24, TXT if i < 6 else DIM, 21)

    # -- main loop --------------------------------------------------------------------------------------
    def run(self) -> int:
        pygame.init()
        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SWSURFACE)
        pygame.display.set_caption("Go2 - FPV / drive / tricks / follow")
        threading.Thread(target=self.connect_bg, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        threading.Thread(target=self.vision_loop, daemon=True).start()
        if not _SELFTEST:
            threading.Thread(target=self._prepare_voice, daemon=True).start()
        clock = pygame.time.Clock()
        script = (self.selftest_script(follow=self.args.selftest_follow, voice=self.args.selftest_voice,
                                       objects=self.args.selftest_objects)
                  if (self.args.selftest or self.args.selftest_follow or self.args.selftest_voice
                      or self.args.selftest_objects) else None)
        t0 = time.time()
        try:
            while self.running:
                if script:
                    script(time.time() - t0, screen)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.running = False
                    elif ev.type == pygame.KEYDOWN:
                        self.on_key(ev.key)
                    elif ev.type == pygame.KEYUP:
                        self.held.discard(ev.key)
                        if ev.key == VOICE_KEY:
                            self.stop_listening()
                    elif ev.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
                        self.held.clear()  # KEYUPs never arrive once focus is lost: don't keep walking
                        self.cancel_listening()
                        self.stop_follow("window lost focus")
                        self.say("window lost focus: stopped", WARN)
                self.drain_voice()
                self.update_velocity()
                self.draw(screen)
                pygame.display.flip()
                clock.tick(30)
        finally:
            print("Stopping and disconnecting ...", flush=True)
            self.stop_evt.set()
            self.abort.set()
            self.following = False
            self.desired = (0.0, 0.0, 0.0)
            if self.robot is not None:
                try:
                    self.robot.stop_move()
                except Exception:  # noqa: BLE001
                    pass
                self.robot.close()
            pygame.quit()
        if self.args.selftest_follow:
            return self.selftest_follow_verdict()
        if self.args.selftest_voice:
            return self.selftest_voice_verdict()
        if self.args.selftest_objects:
            return _selftest_objects_verdict(self)
        return self.selftest_verdict() if self.args.selftest else 0

    # -- headless self-tests (development only) ---------------------------------------------------------
    def selftest_script(self, follow: bool = False, voice: bool = False, objects: bool = False):
        K = pygame
        if objects:  # toggle the overlay on, look at it, toggle it off
            steps = [(0.5, K.KEYDOWN, K.K_o), (0.7, K.KEYUP, K.K_o), (3.6, K.KEYDOWN, K.K_o), (3.8, K.KEYUP, K.K_o),
                     (4.6, K.KEYDOWN, K.K_ESCAPE)]
            shot_at = 2.6
        elif voice:  # hold V briefly for each canned transcript (see main(): FakeMic / FakeSTT)
            steps = []
            for i in range(6):
                steps += [(0.5 + i * 0.7, K.KEYDOWN, K.K_v), (0.8 + i * 0.7, K.KEYUP, K.K_v)]
            steps.append((5.8, K.KEYDOWN, K.K_ESCAPE))
            shot_at = 3.6
        elif follow:
            steps = [(0.5, K.KEYDOWN, K.K_t), (0.7, K.KEYUP, K.K_t), (0.9, K.KEYDOWN, K.K_y), (1.1, K.KEYUP, K.K_y),
                     (4.5, K.KEYDOWN, K.K_SPACE), (7.0, K.KEYDOWN, K.K_ESCAPE)]
            shot_at = 3.5
        else:
            steps = [  # (seconds, event type, key)
                (0.6, K.KEYDOWN, K.K_w), (2.2, K.KEYUP, K.K_w),          # arms, then drives forward
                (2.5, K.KEYDOWN, K.K_n), (2.7, K.KEYDOWN, K.K_w), (2.9, K.KEYUP, K.K_w),  # dance pending, cancelled by W
                (3.1, K.KEYDOWN, K.K_n), (3.3, K.KEYDOWN, K.K_y),        # dance confirmed
                (3.5, K.KEYDOWN, K.K_q), (3.7, K.KEYUP, K.K_q),          # driving locked out while dancing
                (3.9, K.KEYDOWN, K.K_SPACE),                              # e-stop clears the lockout
                (4.1, K.KEYDOWN, K.K_a), (5.6, K.KEYUP, K.K_a),          # re-arms, turns left
                (6.2, K.KEYDOWN, K.K_ESCAPE),
            ]
            shot_at = 5.0
        done = set()
        state = {"shot": False, "marked": False}

        def script(t, screen):
            for i, (when, typ, key) in enumerate(steps):
                if t >= when and i not in done:
                    done.add(i)
                    pygame.event.post(pygame.event.Event(typ, key=key, mod=0, unicode="", scancode=0))
            if t >= shot_at and not state["shot"]:
                state["shot"] = True
                pygame.image.save(screen, "/tmp/go2_selftest_objects.png" if objects else
                                  "/tmp/go2_selftest_voice.png" if voice else
                                  "/tmp/go2_selftest_follow.png" if follow else "/tmp/go2_selftest.png")
            if objects and t >= 3.0 and self._obj_snapshot is None and self.objects:
                self._obj_snapshot = list(self.objects[1])   # what the overlay held just before it was switched off
            if follow and t >= 5.0 and not state["marked"] and isinstance(self.robot, FakeRobot):
                state["marked"] = True  # 0.5 s after Space: nothing should move the dog from here on
                self._log_len_after_stop = len(self.robot.log)

        return script

    def selftest_verdict(self) -> int:
        log = self.robot.log if isinstance(self.robot, FakeRobot) else []
        sports = [e[1] for e in log if e[0] == "sport"]
        moves = [e for e in log if e[0] == "move"]
        checks = {
            "sport order BalanceStand, Dance1, StopMove, BalanceStand": sports == ["BalanceStand", "Dance1", "StopMove", "BalanceStand"],
            "drove forward at 0.4": ("move", 0.4, 0.0, 0.0) in moves,
            "turned left at 0.8 rad/s": ("move", 0.0, 0.0, 0.8) in moves,
            "no sideways move while dance lockout": not any(m[2] != 0 for m in moves),
            "stopped after key release": ("stop_move",) in log,
        }
        print("\nSELFTEST command log:", log)
        for name, ok in checks.items():
            print(("  PASS  " if ok else "  FAIL  ") + name)
        return 0 if all(checks.values()) else 1

    def selftest_voice_verdict(self) -> int:
        return _selftest_voice_verdict(self)

    def selftest_follow_verdict(self) -> int:
        log = self.robot.log if isinstance(self.robot, FakeRobot) else []
        sports = [e[1] for e in log if e[0] == "sport"]
        moves = [e for e in log if e[0] == "move"]
        after = log[self._log_len_after_stop:] if self._log_len_after_stop is not None else None
        checks = {
            "balanced first, then e-stop (BalanceStand, StopMove)": sports == ["BalanceStand", "StopMove"],
            "walked toward the person (vx > 0.3)": any(m[1] > 0.3 for m in moves),
            "turned RIGHT toward a person who is right of centre (yaw < 0)": any(m[3] < -0.3 for m in moves),
            "never strafed": all(m[2] == 0 for m in moves),
            "no walking after Space": after is not None and not any(e[0] == "move" for e in after),
            "follow ended": not self.following,
        }
        print("\nSELFTEST-FOLLOW command log:", log)
        for name, ok in checks.items():
            print(("  PASS  " if ok else "  FAIL  ") + name)
        return 0 if all(checks.values()) else 1


VOICE_TEST_LINES = ["say hello", "dance two", "follow me", "stop", "what is the weather", "stop following"]


def _selftest_voice_verdict(app: "App") -> int:
    log = app.robot.log if isinstance(app.robot, FakeRobot) else []
    sports = [e[1] for e in log if e[0] == "sport"]
    labels = [lab for _, lab in app.heard_log]
    checks = {
        "'say hello' -> Hello, 'dance two' -> Dance2, 'stop' -> StopMove (and nothing else)": sports == ["Hello", "Dance2", "StopMove"],
        "'follow me' did NOT start following (needs the Y key)": app.follow_starts == 0,
        "'what is the weather' matched nothing": labels[4] is None,
        "every transcript was handled": labels == ["Hello", "Dance 2", "follow the nearest person", "STOP", None, "stop following"],
        "'stop' cleared the pending follow confirm": app.pending is None,
        "not following at the end": not app.following,
    }
    print("\nSELFTEST-VOICE heard:", app.heard_log)
    print("SELFTEST-VOICE command log:", log)
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


def _selftest_objects_verdict(app: "App") -> int:
    snap = app._obj_snapshot or []
    names = {follow_mod.COCO_CLASSES[d[5]] for d in snap}
    checks = {
        "overlay produced detections while on (>= 5 boxes)": len(snap) >= 5,
        "found the tv (people this small in a 16:9 frame are below what YOLOX-tiny reliably sees)": "tv" in names,
        "found furniture too (chair or dining table)": bool(names & {"chair", "dining table"}),
        "pressing O again switched it off and cleared the boxes": not app.objects_on and app.objects is None,
        "objects mode never moved the dog": not any(e[0] in ("move", "sport") for e in getattr(app.robot, "log", [])),
    }
    print("\nSELFTEST-OBJECTS saw:", follow_mod.summarize(snap))
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


def _objects_test_canvas() -> np.ndarray | None:
    """1280x720 frame made from the COCO living-room photo (letterboxed, grey sides)."""
    import cv2

    path = "/tmp/go2test/000000000139.jpg"
    if not os.path.exists(path):
        print(f"missing {path}")
        return None
    img = cv2.imread(path)[:, :, ::-1]
    h, w = img.shape[:2]
    s = 720 / h
    big = cv2.resize(img, (int(w * s), 720))
    canvas = np.full((720, 1280, 3), 60, np.uint8)
    x0 = (1280 - big.shape[1]) // 2
    canvas[:, x0 : x0 + big.shape[1]] = big
    return canvas


def _follow_test_canvas() -> np.ndarray | None:
    """A 1280x720 frame with the COCO 'skier' person small and right of centre (err = +0.2, height ~0.23)."""
    import cv2

    path = "/tmp/go2test/000000000785.jpg"
    if not os.path.exists(path):
        print(f"missing {path}")
        return None
    small = cv2.resize(cv2.imread(path)[:, :, ::-1], (320, 212))
    canvas = np.full((720, 1280, 3), 200, np.uint8)
    canvas[300:512, 701:1021] = small
    return canvas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default=os.environ.get("ROBOT_IP", "192.168.12.1"))
    p.add_argument("--demo", action="store_true", help="fake robot + synthetic video, no dog needed")
    p.add_argument("--fetch-model", action="store_true", help="download the object detector and voice model (needs internet), then exit")
    p.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selftest-follow", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selftest-voice", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selftest-objects", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--linear", type=float, default=LINEAR, help="forward/sideways speed, m/s")
    p.add_argument("--angular", type=float, default=ANGULAR, help="turn speed, rad/s")
    p.add_argument("--follow-speed", type=float, default=0.35, help="max forward speed while following, m/s")
    p.add_argument("--stt", choices=["local", "elevenlabs"], default=os.environ.get("GO2_STT", "local"),
                   help="speech-to-text for the V key: 'local' = offline Whisper (default), 'elevenlabs' = cloud (needs internet + key)")
    p.add_argument("--whisper-model", default=os.environ.get("GO2_WHISPER_MODEL", "base.en"),
                   help="local Whisper model (base.en is fast and accurate for short commands; small.en is slower)")
    p.add_argument("--motion-mode", choices=["normal", "ai", "mcf"], default=None,
                   help="optional, UNTESTED: switch the dog's motion controller at connect (DimOS notes 'mcf' is the one that traverses stairs)")
    args = p.parse_args()
    if args.fetch_model:  # everything that needs the internet, done once, so the dog's Wi-Fi is enough afterwards
        rc = 0
        if follow_mod is None:
            print(f"follow.py couldn't be loaded: {_FOLLOW_ERR}")
            rc = 1
        elif follow_mod.model_present():
            print("object/person detector: already downloaded")
        else:
            follow_mod.fetch_model()
        if voice_mod is None:
            print(f"voice.py couldn't be loaded: {_VOICE_ERR}")
            rc = 1
        elif voice_mod.whisper_model_path(args.whisper_model):
            print(f"voice model '{args.whisper_model}': already downloaded")
        else:
            voice_mod.fetch_whisper_model(args.whisper_model)
        return rc
    image = None
    if args.selftest_follow:
        image = _follow_test_canvas()
        if image is None:
            return 2
    if args.selftest_objects:
        image = _objects_test_canvas()
        if image is None:
            return 2
    app = App(args, demo_image=image)
    if args.selftest_voice:
        app.mic, app.stt = FakeMic(), FakeSTT(VOICE_TEST_LINES)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
