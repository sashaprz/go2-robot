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
import math
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
    import obstacles  # lidar point cloud -> dog-frame points (obstacles.py), used by heel --heel-lidar
except Exception as _oe:  # noqa: BLE001
    obstacles = None
    _OBST_ERR = str(_oe)

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
CALIB_KEY = pygame.K_j    # teach the dog how far away you are (two marked spots), for heel
CAL_SPOTS = (1.2, 2.4)    # metres straight ahead of the dog's nose (where the camera is) to stand for calibration
CAL_LIDAR_SPOTS = 3       # calibrating by lidar: this many spots at clearly different distances
CAL_SECS = 1.5            # how long to watch you at each spot
HEEL_KEY = pygame.K_h     # walk at the person's side (asks for Y, like follow)
VOICE_KEY = pygame.K_v   # hold to talk
OBJECTS_KEY = pygame.K_o  # toggle the object-detection overlay
UPRIGHT_KEY = pygame.K_u  # stand on the back legs (asks for Y) / come back down
LISTEN_KEY = pygame.K_l   # toggle always-listening (wake word)
WAKE_WINDOW = 6.0         # seconds the wake word stays "open" after "ernest" on its own


def _reply_code(reply):
    """The status code in a dog reply (0 = accepted), or None when there is no recognisable code (fake robot, odd shape)."""
    try:
        return int(reply["data"]["header"]["status"]["code"])
    except Exception:  # noqa: BLE001
        return None


BOX_BEAT = 1.2            # seconds per side of the box-step square
BOX_SPEED = 0.6           # box-step speed as a fraction of --linear (0.6 x 0.4 m/s = ~0.3 m per side)


def box_step_cmd(t: float, speed: float) -> tuple[float, float, float]:
    """Velocity (vx, vy, vyaw) for the box step, t seconds after it began: forward, right, back, left, repeat.
    Each side eases in and out (sin) so the dog steps rather than lurches; the four sides cancel out."""
    beat, frac = divmod(max(t, 0.0) / BOX_BEAT, 1.0)
    v = speed * (math.pi / 2) * math.sin(math.pi * frac)
    return [(v, 0.0, 0.0), (0.0, -v, 0.0), (-v, 0.0, 0.0), (0.0, v, 0.0)][int(beat) % 4]

# EDIT FREELY. Timed lists of the sport commands above; each step waits that command's busy time.
ROUTINES = {"greeting": ["StandUp", "BalanceStand", "Hello", "Content", "WiggleHips", "Sit"]}
# Deliberately absent: flips, handstand, bound. They can hurt the robot and most aren't supported on an Air.

WIN_W, WIN_H = 1024, 800
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

    def motion_mode(self) -> str:
        """The dog's current motion controller ('normal' / 'ai' / 'mcf'), as the motion switcher reports it."""
        import json as _json

        coro = self.c.conn.datachannel.pub_sub.publish_request_new(self._topic["MOTION_SWITCHER"], {"api_id": 1001})
        resp = asyncio.run_coroutine_threadsafe(coro, self.c.loop).result(timeout=8)
        return str(_json.loads(resp["data"]["data"]).get("name"))

    def sport_api(self, api_id: int, param=None) -> None:
        """A sport command by raw numeric id, with an optional parameter (e.g. {"data": True})."""
        req = {"api_id": api_id}
        if param is not None:
            req["parameter"] = param
        coro = self.c.conn.datachannel.pub_sub.publish_request_new(self._topic["SPORT_MOD"], req)
        return asyncio.run_coroutine_threadsafe(coro, self.c.loop).result(timeout=8)

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

    def on_lidar(self, cb) -> None:
        """cb(points_world (N, 3) float32, pose (x, y, z, yaw)) for every lidar message. Switches the dog's lidar on
        (DimOS doesn't). If this dog sends no lidar, cb is simply never called."""
        pose: dict = {}
        self._subs.append(self.c.odom_stream().subscribe(
            lambda p: pose.update(v=(p.position.x, p.position.y, p.position.z, p.yaw))))

        def handle(msg):
            try:
                pts = np.asarray(msg["data"]["data"]["points"], dtype=np.float32).reshape(-1, 3)
            except Exception:  # noqa: BLE001 - odd/partial message: just skip it
                return
            if "v" in pose:
                cb(pts, pose["v"])

        self._subs.append(self.c.raw_lidar_stream().subscribe(handle))
        self.c.loop.call_soon_threadsafe(
            self.c.conn.datachannel.pub_sub.publish_without_callback, self._topic["ULIDAR_SWITCH"], "on")

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
        self.refuse: set[int] = set()       # sport api ids this fake dog answers with an error code (tests the fallbacks)

    def _add(self, entry: tuple) -> None:
        if not self.log or self.log[-1] != entry:  # collapse the 20 Hz repeats
            self.log.append(entry)

    def sport(self, name: str) -> None:
        self._add(("sport", name))

    def motion_mode(self) -> str:
        return "fake"

    def sport_api(self, api_id: int, param=None):
        self._add(("api", api_id, param))
        return {"data": {"header": {"status": {"code": 7 if api_id in self.refuse else 0}}}}

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

    def on_lidar(self, cb) -> None:
        """A pretend lidar (self-tests): a person-sized blob of points 1.8 m ahead and 0.45 m to the right, 8 times a second."""
        rng = np.random.default_rng(3)
        blob = np.stack((1.8 + rng.uniform(-.15, .15, 40), -0.45 + rng.uniform(-.15, .15, 40), rng.uniform(0.2, 1.7, 40)), axis=1)

        def gen():
            while not self._halt.is_set():
                cb(blob.astype(np.float32), (0.0, 0.0, 0.32, 0.0))
                time.sleep(0.125)

        threading.Thread(target=gen, daemon=True).start()

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
        self.voice_move = None              # {"cmd", "dur", "start", "created", "label"} while a spoken move runs
        self._voice_gen = 0                 # bumped per transcription so a timed-out one can be ignored
        self._voice_deadline = 0.0
        self.heard_log: list = []           # (transcript, matched label) - used by the self-test
        self.follow_starts = 0              # how many times follow actually started - self-test
        self.all_said: list = []            # everything said (self-tests)
        self._lidar_times: deque = deque(maxlen=40)    # for the lidar rate readout
        self.cal_events: list = []          # (self-test) what calibration did
        self.cal = None                     # calibration in progress: {step, phase, boxes, samples, until, last}
        self._det_times: deque = deque(maxlen=30)   # when the detector last finished (for the fps readout)
        self._lock_warned = 0.0             # last time we said we were ignoring a stranger
        self._lidar_on = False              # has the lidar feed been started?
        self._lidar_check = None            # when to report whether lidar data is arriving
        self.lidar = None                   # (time, points_world, pose) - the latest lidar message, when --heel-lidar
        self.follow_sources: set = set()    # which sensors the heel controller has used (self-test)
        self.follow_mode = "follow"          # "follow" (chase them) or "heel" (walk at their side); both run through self.follower
        # object detection overlay
        self.objects_on = False
        self.objects = None                 # (timestamp, [(x1, y1, x2, y2, score, class_id)], frame_shape)
        self._obj_snapshot = None           # self-test bookkeeping
        # always-listening (wake word), standing on the back legs
        self.listener = None
        self.wake_until = 0.0               # the wake word ("ernest") alone opens this window
        self.last_heard = None              # (time, text, routing verdict) for the status bar
        self.ear = "off"                    # "off" | "on" | "error"
        self.upright = False
        self.upright_api = args.upright_api   # the back-leg command that worked last (or the one to try first)
        self.box_step = False               # True from "box step" until "stop"
        self.box_dancing = False            # True once it is up on the back legs: the box-step square runs
        self.box_t0 = None                  # when the current box-step square started
        self.ambient_log: list = []         # (text, verdict) - self-test

    def is_selftest(self) -> bool:
        return any(k.startswith("selftest") and v for k, v in vars(self.args).items())

    # -- feedback ---------------------------------------------------------------------------------------
    def messages_text(self) -> list:
        return [m[1] for m in self.messages] + list(getattr(self, "all_said", []))

    def say(self, text: str, color=TXT) -> None:
        if _SELFTEST:
            self.all_said.append(text)
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
            if self.args.demo or self.is_selftest():
                r = FakeRobot(self.demo_image)
                if self.args.selftest_listen:
                    # this fake dog only knows the older back-leg id (exercises the fallback); GO2_TEST_REFUSE=2050,1050 = no back legs
                    r.refuse = {int(x) for x in os.environ.get("GO2_TEST_REFUSE", "2050").split(",") if x}
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
            if getattr(self.args, "cal_loaded", None):
                self.say(f"heel: using your saved camera calibration ({self.args.cal_loaded[0] * 100:.0f} cm high, tilted {self.args.cal_loaded[1]:.1f} deg). J recalibrates")
            try:
                self.say(f"dog motion mode: {r.motion_mode()}")
            except Exception as e:  # noqa: BLE001
                self.say(f"couldn't read the dog's motion mode: {e}", WARN)
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
        if self.upright:
            self.say("come down to four legs first (U / say 'come down')", WARN)
            return
        cfg = follow_mod.FollowConfig(max_forward=self.args.follow_speed)
        self.follower = follow_mod.Follower(cfg)
        self.follower.reset()
        self.follow_res = None
        self.voice_move = None
        self.follow_mode = "follow"
        self.following = True
        self.follow_starts += 1
        self.say("> following the nearest person (Space / T / any drive key stops)", GOOD)
        self.ensure_detector()

    # -- calibration: teach the dog how far away you are (camera height / tilt), no tape-and-flags needed ------
    def start_calibration(self) -> None:
        if not self.follow_available():
            return
        if self.robot is None or self.latest is None:
            self.say("not connected / no video yet", WARN)
            return
        self.stop_follow("calibrating")
        self.cal = {"step": 0, "phase": "wait", "mode": None, "total": CAL_LIDAR_SPOTS, "boxes": [], "clouds": [], "samples": [],
                    "until": 0.0, "last": None, "h": 0}
        self.ensure_detector()
        if obstacles is not None:
            self.ensure_lidar()
        self.say("CALIBRATION: stand about 2 m in front of the dog, in the open (a metre from walls and furniture), feet in the picture, "
                 "then press J. Space cancels.", WARN)

    def _lidar_fresh(self) -> bool:
        lid = self.lidar
        return lid is not None and time.time() - lid[0] < 1.0

    def calibration_press(self) -> None:
        cal = self.cal
        if cal is None or cal["phase"] != "wait":
            return
        if cal["mode"] is None:                          # first press: use the lidar if the dog is sending it, else the tape
            if obstacles is not None and self._lidar_fresh():
                cal["mode"], cal["total"] = "lidar", CAL_LIDAR_SPOTS
                self.say(f"calibration by LIDAR (no tape): {CAL_LIDAR_SPOTS} spots, each at a clearly different distance (at least 0.5 m apart, "
                         "between 1 and 3.5 m)")
            else:
                cal["mode"], cal["total"] = "tape", len(CAL_SPOTS)
                self.say("no lidar data from the dog, so calibration needs a TAPE. " + f"Stand {CAL_SPOTS[0]} m straight in front of its nose, then press J", WARN)
                return
        cal["phase"], cal["boxes"], cal["clouds"], cal["until"] = "collect", [], [], time.time() + CAL_SECS
        self.say("measuring: stand still ...")

    def update_calibration(self, now: float) -> None:
        cal = self.cal
        if cal is None or cal["phase"] != "collect":
            return
        if self._lidar_fresh() and obstacles is not None and len(cal["clouds"]) < 12 and (not cal["clouds"] or cal["clouds"][-1][0] != self.lidar[0]):
            cal["clouds"].append((self.lidar[0], obstacles.to_dog_frame(self.lidar[1], *self.lidar[2])))
        if now < cal["until"]:
            return
        boxes, cal["phase"] = cal["boxes"], "wait"
        if len(boxes) < 5:
            self.say(f"calibration: only saw you in {len(boxes)} frames (need 5+). Is the detector loaded and your whole body in view? Press J to retry", WARN)
            return
        h, w = boxes[0][1]
        rows = sorted(b[0][3] for b in boxes)
        row = rows[len(rows) // 2]                                   # median row of the feet
        if row > h - 6:
            self.say("calibration: your feet are cut off at the bottom of the picture. Step back so they show, then press J", WARN)
            return
        if cal["mode"] == "lidar":
            xs = sorted((b[0][0] + b[0][2]) / 2 for b in boxes)
            cam = follow_mod.Camera()
            bearing = math.atan((cam.cx * w / cam.width - xs[len(xs) // 2]) / (cam.fx * w / cam.width))     # + = left of the picture's centre
            ranges = [r for r in (obstacles.refine_range(c[1], (cam.x_off, 0.0), bearing, 2.5, window=2.0) for c in cal["clouds"]) if r is not None]
            if len(ranges) < 3:
                self.say(f"calibration: the lidar gave {len(cal['clouds'])} scans but only {len(ranges)} showed a person-sized blob where the camera sees you. "
                         "Stand in the open (a metre from walls and furniture), 1 to 3.5 m ahead, then press J. "
                         + ("(No lidar scans arrived at all: is the lidar on?)" if not cal["clouds"] else ""), WARN)
                return
            dist = sorted(ranges)[len(ranges) // 2] * math.cos(bearing)     # straight ahead of the lens, as the solver wants
            if any(abs(dist - d) < 0.5 for _, d in cal["samples"]):
                self.cal_events.append(("same_distance",))
                self.say(f"that's about the same distance as a spot already recorded ({dist:.2f} m by the lidar). Move at least 0.5 m closer or "
                         "further, then press J", WARN)
                return
            cal["samples"].append((row, dist))
            self.say(f"spot {len(cal['samples'])} of {cal['total']}: the lidar puts you {dist:.2f} m ahead (feet at picture row {row:.0f}).")
        else:
            cal["samples"].append((row, CAL_SPOTS[cal["step"]]))
        self.cal_events.append(("spot", row))
        cal["h"] = h
        cal["step"] += 1
        if cal["step"] < cal["total"]:
            self.say(("Now stand at a clearly different distance" if cal["mode"] == "lidar" else f"Now stand {CAL_SPOTS[cal['step']]} m straight in front of its nose")
                     + f" (spot {cal['step'] + 1} of {cal['total']}), then press J", WARN)
            return
        if self.finish_calibration(cal["samples"], h):
            self.cal = None
        else:
            cal["step"], cal["samples"] = 0, []

    def finish_calibration(self, samples, frame_h: int) -> bool:
        """Solve for the camera height / tilt from the measurements, apply them now and save them. False = refused."""
        res = follow_mod.solve_camera(samples, frame_h=frame_h)
        self.cal_events.append(("solved" if res else "refused",))
        if res is None:
            self.say("calibration: those two measurements don't fit any camera (a distance measured wrong, or your feet weren't really visible). "
                     f"Nothing was changed. Start again: stand {CAL_SPOTS[0]} m in front of its nose and press J", BAD)
            return False
        z, tilt, rms = res
        self.args.heel_cam_height, self.args.heel_cam_pitch = z, math.degrees(tilt)
        try:
            follow_mod.save_calibration(z, math.degrees(tilt), **({"path": "/tmp/go2_cal_test.json"} if self.is_selftest() else {}))
            saved = "Saved: go2.bat loads it automatically from now on."
        except OSError as e:
            saved = f"(couldn't save it: {e})"
        self.say(f"CALIBRATED: camera {z * 100:.0f} cm off the floor, tilted {math.degrees(tilt):.1f} deg down (fit error {rms * 100:.0f} cm). {saved}", GOOD)
        return True

    def start_heel(self, side: str = "") -> None:
        """Walk at the person's side. Same plumbing as follow (self.follower.step() gives the velocity), different controller."""
        if self.upright:
            self.say("come down to four legs first (U / say 'come down')", WARN)
            return
        side = side or self.args.heel_side
        cam = follow_mod.Camera(z=self.args.heel_cam_height, pitch=math.radians(self.args.heel_cam_pitch))
        self.follower = follow_mod.Heeler(follow_mod.HeelConfig(side=side, max_forward=self.args.heel_speed, cam=cam, lead=self.args.heel_lead, gap=self.args.heel_gap,
                                                             use_lidar=self.args.heel_lidar and obstacles is not None))
        self.follower.reset()
        self.follow_res = None
        self.voice_move = None
        self.follow_mode = "heel"
        self.following = True
        self.follow_starts += 1
        use_lidar = self.args.heel_lidar and obstacles is not None
        if use_lidar:
            self.ensure_lidar()
            self._lidar_check = time.time() + 3.0
        self.say(f"> heeling: walking on your {side} (Space / H / any drive key stops). "
                 + ("Camera finds and follows you; lidar only sharpens the distance" if use_lidar
                    else "Camera only: it keeps ~1.3 m behind your hip"), GOOD)
        self.ensure_detector()

    def _on_lidar(self, pts, pose) -> None:
        self.lidar = (time.time(), pts, pose)
        self._lidar_times.append(self.lidar[0])

    def ensure_lidar(self) -> None:
        """Start the dog's lidar feed the first time heel needs it (not at connect: it is extra traffic on the link)."""
        if self._lidar_on or self.robot is None or not hasattr(self.robot, "on_lidar"):
            return
        self._lidar_on = True
        try:
            self.robot.on_lidar(self._on_lidar)
        except Exception as e:  # noqa: BLE001
            self._lidar_on = False
            self.say(f"couldn't start the lidar: {e}. Heel will use the camera only", WARN)

    def stop_follow(self, reason: str) -> None:
        if not self.following:
            return
        self.following = False
        self.follow_res = None
        self.desired = (0.0, 0.0, 0.0)
        self.say(f"{'heel' if self.follow_mode == 'heel' else 'follow'} stopped: {reason}", WARN)

    def vision_loop(self) -> None:
        """One inference per new frame, shared by person-follow and the object overlay."""
        last_seq = -1
        while not self.stop_evt.is_set():
            det = self.detector
            if not (self.following or self.objects_on or self.cal) or det is None or self.latest is None or self.frame_seq == last_seq:
                time.sleep(0.03)
                continue
            with self.frame_lock:
                frame, last_seq = self.latest, self.frame_seq
            try:
                dets = det.detect_objects(frame)
                cal = self.cal
                if cal is not None:
                    ppl = follow_mod.people(dets, frame.shape[0])
                    if ppl:
                        big = max(ppl, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                        cal["last"] = (time.time(), big[:4])
                        if cal["phase"] == "collect":
                            cal["boxes"].append((big[:4], frame.shape[:2]))
                if self.objects_on:
                    self.objects = (time.time(), dets, frame.shape)
                res = None
                if self.following:
                    kw = {"frame": frame}                                          # the picture: for the clothing-colour lock
                    if self.follow_mode == "heel" and self.args.heel_lidar and obstacles is not None:
                        lid = self.lidar
                        if lid is not None and time.time() - lid[0] < 0.5:      # a stale cloud is worse than none
                            kw["cloud"] = obstacles.to_dog_frame(lid[1], *lid[2])
                    res = self.follower.step(follow_mod.people(dets, frame.shape[0]), frame.shape, **kw)
                    self.follow_sources.add(getattr(self.follower, "source", "camera"))
                    self._det_times.append(time.time())
                    lk = self.follower.lock
                    if lk.label and not lk.announced:
                        lk.announced = True
                        self.say(f"locked onto: {lk.label}. It will only follow someone who looks like this", GOOD)
                    if lk.rejected and time.time() - self._lock_warned > 6:
                        self._lock_warned = time.time()
                        self.say("ignoring another person who isn't dressed like the one it locked onto", DIM)
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
                    if not self.args.no_listen:
                        self.start_listener()
        except Exception as e:  # noqa: BLE001
            self.say(f"voice model failed to load: {e}", BAD)

    # -- always-listening (wake word) -------------------------------------------------------------------
    def start_listener(self) -> None:
        if self.listener is not None:
            return
        if voice_mod is None or self.stt is None or self.args.stt != "local":
            self.say("always-listening needs the offline model (default --stt local)", WARN)
            return
        try:
            self.listener = voice_mod.AlwaysListener(self.stt, lambda t: self.voice_q.put(("ambient", t, -1)),
                                                     on_error=self._ear_error, log_dir=self.args.voice_log)
            self.listener.start()
        except Exception as e:  # noqa: BLE001
            self.listener, self.ear = None, "error"
            self.say(f"always-listening unavailable: {e}", WARN)
            return
        self.ear = "on"
        self.say(f'always listening: say "{self.args.wake_word}" + a command. A bare "stop" works without it.', GOOD)

    def _ear_error(self, msg: str) -> None:
        self.ear = "error"
        self.say(f"ear: {msg}", BAD)

    def stop_listener(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self.ear = "off"

    def toggle_listener(self) -> None:
        if self.listener is not None:
            self.stop_listener()
            self.say("always-listening OFF (hold V to talk)", WARN)
        elif self.voice_ready():
            self.start_listener()

    def handle_ambient(self, text: str) -> None:
        """An utterance from the always-on mic. Only acts on the wake word (+ command), a bare 'stop', or a bare
        'yes' while something waits for confirmation. Everything else is conversation and is ignored."""
        now = time.time()
        pending = bool(self.pending and now < self.pending[2])
        action, rest = voice_mod.route_utterance(text, self.args.wake_word, wake_active=now < self.wake_until,
                                                 confirm_pending=pending)
        self.ambient_log.append((text, action))
        self.last_heard = (now, text, action)
        if action == "ignore":
            return
        if action == "wake":
            self.wake_until = now + WAKE_WINDOW
            self.say(f'{self.args.wake_word}: yes? say a command', GOOD)
            return
        self.wake_until = 0.0
        if action == "confirm":
            self.confirm_pending()
        else:                                   # "command" or a bare "stop"
            self.handle_heard(rest)

    def confirm_pending(self) -> None:
        if not self.pending:
            return
        label, action, deadline = self.pending
        self.pending = None
        if time.time() < deadline:
            action()
        else:
            self.say(f"confirmation for {label} timed out", WARN)

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
            if kind == "ambient":
                self.handle_ambient(payload)
                continue
            if gen != self._voice_gen:
                continue              # a result from a transcription that already timed out
            if kind == "error":
                self.say(f"voice: {payload}", BAD)
            else:
                self.handle_heard(payload)

    def start_voice_move(self, intent) -> None:
        """'walk forward', 'turn left 90 degrees', ...: a short timed move. Spoken amounts are capped (see voice.py)."""
        cmd, secs = voice_mod.plan_motion(intent, self.args.linear, self.args.angular)
        self.stop_follow("voice move")
        self.abort.set()   # a running routine must not keep issuing tricks while we walk
        self.pending = None
        self.voice_move = {"cmd": cmd, "dur": secs, "start": None, "created": time.time(), "label": intent.label}
        self.say(f"> {intent.label} for {secs:.1f} s   (say 'stop' / Space / any key cancels)")

    def handle_heard(self, text: str) -> None:
        """Turn a transcript into the same actions the keys trigger. Voice is push-to-talk, so it is deliberate:
        tricks, timed moves and FOLLOW all start directly. 'stop' always wins."""
        if self.pending and text and voice_mod.is_confirm(text) and len(text.split()) <= 3:
            self.confirm_pending()                       # a spoken "yes" answers a pending confirmation
            return
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
                self.pending = None
                self.voice_move = None
                self.start_follow()
        elif intent.kind == "heel":
            if self.following and self.follow_mode == "heel" and not intent.arg:
                self.say("already heeling (say 'stop' to end it)")
            elif self.follow_available():
                self.pending = None
                self.voice_move = None
                self.stop_follow("switching")
                self.start_heel(intent.arg)
        elif intent.kind == "move":
            self.start_voice_move(intent)
        elif intent.kind == "upright":
            if intent.arg == "on":
                self.request_upright()                   # asks for a spoken "yes" (or Y): this one can fall
            elif self.upright:
                self.stop_upright("voice")
            else:
                self.say("already on four legs")
        elif intent.kind == "box_step":
            self.start_box_step()
        elif intent.kind == "sport":
            label, wait = self.lookup[intent.arg]
            self.run_trick(label, intent.arg, wait)
        elif intent.kind == "routine":
            self.run_routine(intent.arg)

    # -- standing on the back legs (Unitree "WalkUpright") ----------------------------------------------
    def request_upright(self) -> None:
        if self.upright:
            self.say("already standing on the back legs")
        elif self.robot is None:
            self.say("not connected yet", WARN)
        else:
            self.pending = ("Stand on the BACK LEGS (it can fall: soft floor, clear space, spotter)", self.start_upright,
                            time.time() + CONFIRM_SECS)
            self.say("stand on the back legs? say 'yes' or press Y within 4 s", WARN)

    def start_upright(self) -> None:
        if self.upright or self.robot is None:
            return
        self.stop_follow("back-leg stand")
        self.voice_move = None
        was_armed = self.armed
        self.upright, self.armed = True, True             # (never send BalanceStand while up: it could drop the dog)
        self.busy_until = time.time() + 5.0
        self.say("> standing on the back legs. Say 'come down' / press U to return to four legs", WARN)
        self.pool.submit(self._upright_send, True, was_armed)

    def stop_upright(self, why: str = "") -> None:
        if not self.upright:
            return
        self.upright, self.armed = False, False
        self.box_step = self.box_dancing = False           # coming down ends a box-step session
        self.voice_move, self.desired = None, (0.0, 0.0, 0.0)
        self.busy_until = time.time() + 5.0
        self.say("> coming down to four legs" + (f" ({why})" if why else ""), WARN)
        self.pool.submit(self._upright_send, False, True)

    def _upright_send(self, on: bool, was_armed: bool) -> None:
        try:
            if on and not was_armed:
                self.robot.sport("BalanceStand")          # start from a balanced stand
                time.sleep(1.5)
            order = [self.upright_api] + [a for a in (2050, 1050) if a != self.upright_api]
            tried = []
            for api in (order if on else order[:1]):      # going up tries the other id if the dog refuses the first
                reply = None
                try:
                    reply = self.robot.sport_api(api, {"data": on})
                    code, err = _reply_code(reply), None
                except Exception as e:  # noqa: BLE001
                    code, err = None, e
                ok = err is None and code in (0, None)
                tried.append(f"{api}: " + (type(err).__name__ if err else f"code {code}"))
                self.say(f"back-leg {'stand' if on else 'release'} (api {api}): "
                         + (f"error {type(err).__name__} {err}" if err else f"the dog replied code {code}") + ("" if ok else "  <- refused"),
                         GOOD if ok else WARN)
                if ok:
                    self.upright_api = api
                    return
            try:
                mode = self.robot.motion_mode()
            except Exception:  # noqa: BLE001
                mode = "unknown"
            raise RuntimeError(f"the dog (motion mode: {mode}) refused every back-leg command (" + "; ".join(tried) + ")")
        except Exception as e:  # noqa: BLE001
            self.say(f"back-leg stand {'on' if on else 'off'} failed: {e}", BAD)
            if on:
                self.upright = False
                self.busy_until = time.time()
                if self.box_step:                          # the dog can't do it: still dance, on four legs
                    self.say("> box step continues on FOUR legs (the back-leg stand isn't available on this dog)", WARN)

    # -- "box step": up on the back legs, then step in a square until "stop" -------------------------------
    def start_box_step(self) -> None:
        if self.robot is None:
            self.say("not connected yet", WARN)
        elif self.box_step:
            self.say("box step is already running (say 'stop' to end it)")
        else:
            self.stop_follow("box step")
            self.voice_move, self.pending = None, None
            self.box_step, self.box_dancing, self.box_t0 = True, False, None
            self.say("> box step: up on the back legs, then it steps in a square. Say 'stop' to end it. (It can fall: keep clear.)", WARN)
            self.pool.submit(self._box_step_work)

    def _box_step_work(self) -> None:
        wait = self.args.box_wait
        try:
            if not self.upright:
                self.armed = False
                self.busy_until = time.time() + wait + 1.5
                self.send("StandUp")
                time.sleep(wait)
                if not self.box_step:                    # 'stop' arrived while it was getting up
                    return
                self.send("BalanceStand")
                self.armed = True
                time.sleep(min(1.0, wait))
                if not self.box_step:
                    return
                self.start_upright()                     # onto the back legs; the dance waits out its 5 s busy time
            self.box_dancing = True
        except Exception as e:  # noqa: BLE001 - a dropped connection: say so
            self.say(f"box step: {e}", BAD)

    # -- actions ----------------------------------------------------------------------------------------
    def run_trick(self, label: str, name: str, wait: float) -> None:
        if self.upright:  # poses/tricks while balanced on two legs are undefined: make the human come down first
            self.say(f"not while standing on the back legs: come down first (U / say 'come down'), then {label}", WARN)
            return
        self.voice_move = None
        self.box_step = self.box_dancing = False    # a pose/trick you asked for beats an unfinished box-step sequence
        self.stop_follow(f"{label} pressed")
        self.say(f"> {label}")
        self.busy_until = time.time() + wait
        self.armed = name == "BalanceStand"  # after any other pose/trick, re-balance before driving
        self.pool.submit(self.send, name)

    def run_routine(self, rname: str) -> None:
        if self.upright:
            self.say("not while standing on the back legs: come down first (U / say 'come down')", WARN)
            return
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
        self.voice_move = None
        self.busy_until = 0.0
        self.desired = (0.0, 0.0, 0.0)
        self.stop_follow("SPACE")
        if self.cal is not None:
            self.cal = None
            self.say("calibration cancelled", WARN)
        self.say("STOP", BAD)
        self.pool.submit(self.send, "StopMove")
        self.box_step, self.box_dancing, self.box_t0 = False, False, None    # 'stop' ends a box-step session (dog stays as it is)

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
        elif key == CALIB_KEY:
            if self.cal is None:
                self.start_calibration()
            else:
                self.calibration_press()
        elif key == HEEL_KEY:
            if self.following:
                self.stop_follow("H pressed")
            elif self.follow_available():
                self.pending = ("Heel: walk at your side (NO obstacle avoidance)", self.start_heel, now + CONFIRM_SECS)
        elif key == VOICE_KEY:
            self.start_listening()
        elif key == OBJECTS_KEY:
            self.toggle_objects()
        elif key == UPRIGHT_KEY:
            self.stop_upright("U pressed") if self.upright else self.request_upright()
        elif key == LISTEN_KEY:
            self.toggle_listener()
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
        if self.voice_move:
            vm = self.voice_move
            if manual:
                self.voice_move = None            # the human always wins; carry on as manual below
                self.say("voice move cancelled (key pressed)", WARN)
            else:
                if vm["start"] is None and now - vm["created"] > 10:
                    self.voice_move = None
                    self.desired = (0.0, 0.0, 0.0)
                    self.say("voice move dropped: the dog was busy for over 10 s", WARN)
                    return
                if not self.ensure_ready(now):
                    self.desired = (0.0, 0.0, 0.0)
                    return
                if vm["start"] is None:
                    vm["start"] = now
                if now - vm["start"] >= vm["dur"]:
                    self.voice_move = None
                    self.desired = (0.0, 0.0, 0.0)
                    self.say(f"{vm['label']}: done")
                    return
                self.desired = vm["cmd"]
                return
        if self._lidar_check and now > self._lidar_check and self.following:
            self._lidar_check = None
            if self.lidar is None:
                self.say("heel: NO lidar data from the dog, so it is heeling on the camera alone", WARN)
            else:
                self.say("heel: lidar is streaming (it sharpens the distance; the camera still decides)", GOOD)
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
        if self.box_step and self.box_dancing and not manual:
            if not self.ensure_ready(now):
                self.desired = (0.0, 0.0, 0.0)
                return
            if self.box_t0 is None:
                self.box_t0 = now
            self.desired = box_step_cmd(now - self.box_t0, self.args.linear * BOX_SPEED)
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
                tx, ty = ox + x1 * sc, max(60, oy + y1 * sc - lh)
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
        if self.cal is not None and self.cal["last"] is not None and now - self.cal["last"][0] < 1.0:   # who the calibration is looking at
            ox, oy, sc = self._xf
            x1, y1, x2, y2 = self.cal["last"][1]
            pygame.draw.rect(screen, WARN, (ox + x1 * sc, oy + y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc), 3)
        res = self.follow_res
        if self.following and res and res[1].box and now - res[0] < 1.0:  # box around the tracked person
            ox, oy, sc = self._xf
            x1, y1, x2, y2 = res[1].box[:4]
            pygame.draw.rect(screen, GOOD if res[1].status != "close enough" else WARN,
                             (ox + x1 * sc, oy + y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc), 3)

        fps = 0.0
        if len(self.frame_times) > 2 and now - self.frame_times[-1] < 2:
            fps = (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-3)
        bar = pygame.Surface((WIN_W, 58), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 150))
        screen.blit(bar, (0, 0))
        bat = f"{self.battery}%" if self.battery is not None else "?"
        self.text(screen, self.state, 12, 8, self.state_color, 24)
        lt = [t for t in self._lidar_times if now - t < 3]
        lidar_txt = (f"    lidar {(len(lt) - 1) / max(lt[-1] - lt[0], 1e-3):.0f}/s" if len(lt) > 2 else "    lidar: no data") if self._lidar_on else ""
        self.text(screen, f"video {fps:.0f} fps    battery {bat}{lidar_txt}", 360, 8, TXT, 24)
        d = self.desired
        status = "BUSY" if now < self.busy_until else ("balancing" if self.armed else "will balance on first drive key")
        self.text(screen, f"{status}    vx {d[0]:+.2f}  vy {d[1]:+.2f}  yaw {d[2]:+.2f}", 640, 8, TXT if any(d) else DIM, 24)
        # second row: the ear (wake word), what it last heard, the back-leg state
        if self.ear == "on":
            ear_txt, ear_col = f'ear: ON, say "{self.args.wake_word}"' + (" ... (listening for a command)" if now < self.wake_until else ""), GOOD
        elif self.ear == "error":
            ear_txt, ear_col = "ear: ERROR (press L to retry)", BAD
        else:
            ear_txt, ear_col = "ear: off (L to turn on, V to talk)", DIM
        self.text(screen, ear_txt, 12, 34, ear_col, 22)
        lh = self.last_heard
        if lh and now - lh[0] < 12:
            tone = {"ignore": DIM, "wake": GOOD, "command": GOOD, "stop": WARN, "confirm": GOOD}.get(lh[2], TXT)
            what = {"ignore": " (ignored: no wake word)", "wake": " (wake word)", "command": "", "stop": " (stop)", "confirm": " (yes)"}.get(lh[2], "")
            self.text(screen, f'heard: "{lh[1][:56]}"{what}', 330, 34, tone, 22)
        if self.upright:
            self.text(screen, "UPRIGHT", 900, 34, WARN, 24)

        banner = None
        if self.voice_state == "listening":
            banner = ("LISTENING ...  release V to send", BAD)
        elif self.voice_state == "transcribing":
            banner = ("transcribing ...", WARN)
        elif self.pending and now < self.pending[2]:
            banner = (f"{self.pending[0]}: press Y or say 'yes' ({self.pending[2] - now:.0f}s)", WARN)
        elif self.box_step:
            banner = ("BOX STEP: " + ("getting up on the back legs ..." if not self.box_dancing else "stepping")
                      + "     say 'stop' to end it", GOOD)
        elif self.voice_move:
            vm = self.voice_move
            banner = (f"VOICE MOVE: {vm['label']}" + ("" if vm["start"] else "  (getting ready ...)")
                      + "     say 'stop' / Space / any key cancels", GOOD)
        elif self.cal is not None:
            c = self.cal
            seen = c["last"] is not None and now - c["last"][0] < 1.0
            step = f"{min(c['step'] + 1, c['total'])} of {c['total']}"
            how = {"lidar": " (lidar, no tape)", "tape": " (tape)"}.get(c["mode"], "")
            ask = ("stand at a clearly different distance, in the open" if c["mode"] == "lidar" and c["step"] else
                   "stand ~2 m in front of the dog, in the open" if c["mode"] in (None, "lidar") else
                   f"stand {CAL_SPOTS[min(c['step'], len(CAL_SPOTS) - 1)]} m in front of the dog's NOSE")
            banner = ((f"CALIBRATING {step}{how}: measuring, stand still ...  ({len(c['boxes'])} frames, {len(c['clouds'])} lidar scans)" if c["phase"] == "collect"
                       else f"CALIBRATION {step}{how}: {ask}, then press J   " + ("[I can see you]" if seen else "[I can't see you yet]")),
                      WARN if not seen else GOOD)
        elif self.following:
            st = res[1].status if res else ("loading detector ..." if self.detector is None else "looking for a person ...")
            dt = [t for t in self._det_times if now - t < 3]
            fps = f"  [detector {(len(dt) - 1) / max(dt[-1] - dt[0], 1e-3):.0f} fps]" if len(dt) > 2 else "  [detector: no frames]"
            banner = ((f"HEELING ({self.follower.cfg.side}): {st}{fps}" if self.follow_mode == "heel"
                       else f"FOLLOWING: {st}{fps}"), GOOD)
        if banner:
            back = pygame.Surface((WIN_W, 40), pygame.SRCALPHA)
            back.fill((0, 0, 0, 150))
            screen.blit(back, (0, 60))
            self.text(screen, banner[0], 12, 68, banner[1], 30)

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
            "FOLLOW  T follow the nearest person, H heel (walk at your side), each then Y.  Again / Space / any drive key stops.  No obstacle avoidance!",
            f"VOICE   say \"{self.args.wake_word}, <command>\" or hold V: \"ready to dance\" (stand up), \"sit\", \"dance two\", \"walk forward\", "
            "\"follow me\", \"heel\", \"box step\".  \"stop\" always works."
            + ("" if self.args.stt == "local" else "  (ElevenLabs: needs internet)"),
            "BACK LEGS  U (then Y), or \"" + self.args.wake_word + ", stand on your back legs\" then \"yes\".  U / \"come down\" returns to four legs.  Can fall: soft floor!",
            "VISION  O toggles labelled boxes for 80 object types.   J = calibrate heel distance (stand at 3 different distances; uses the lidar, no tape).",
            "Click this window so it has keyboard focus.   Esc quits.",
        ]
        for i, line in enumerate(lines):
            self.text(screen, line, 12, VIDEO_H + 10 + i * 24, TXT if i < 7 else DIM, 21)

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
                                       objects=self.args.selftest_objects, voicemove=self.args.selftest_voicemove,
                                       listen=self.args.selftest_listen, heel=self.args.selftest_heel, calib=self.args.selftest_calib)
                  if self.is_selftest() else None)
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
                        self.voice_move = None
                        self.cancel_listening()
                        self.stop_follow("window lost focus")
                        self.say("window lost focus: stopped", WARN)
                if self.listener is not None:
                    self.listener.paused = bool(self.voice_state)   # hold-V push-to-talk takes priority over the ear
                self.drain_voice()
                self.update_calibration(time.time())
                self.update_velocity()
                self.draw(screen)
                pygame.display.flip()
                clock.tick(30)
        finally:
            print("Stopping and disconnecting ...", flush=True)
            self.stop_listener()
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
        if self.args.selftest_voicemove:
            return _selftest_voicemove_verdict(self)
        if self.args.selftest_listen:
            return _selftest_listen_verdict(self)
        if self.args.selftest_heel:
            return _selftest_heel_verdict(self)
        if self.args.selftest_calib:
            return _selftest_calib_verdict(self)
        return self.selftest_verdict() if self.args.selftest else 0

    # -- headless self-tests (development only) ---------------------------------------------------------
    def selftest_script(self, follow: bool = False, voice: bool = False, objects: bool = False, voicemove: bool = False,
                        listen: bool = False, heel: bool = False, calib: bool = False):
        K = pygame
        if listen:  # utterances are injected as if the always-on mic had heard them (see LISTEN_TEST)
            steps = [(19.3, K.KEYDOWN, K.K_ESCAPE)]
            shot_at = 15.0
        elif voicemove:  # six push-to-talk presses, one per canned transcript; 'stop' lands mid-step
            steps = []
            for t in (0.5, 2.6, 3.6, 4.7, 6.6, 7.2):
                steps += [(t, K.KEYDOWN, K.K_v), (t + 0.3, K.KEYUP, K.K_v)]
            steps.append((9.0, K.KEYDOWN, K.K_ESCAPE))
            shot_at = 4.3
        elif objects:  # toggle the overlay on, look at it, toggle it off
            steps = [(0.5, K.KEYDOWN, K.K_o), (0.7, K.KEYUP, K.K_o), (3.6, K.KEYDOWN, K.K_o), (3.8, K.KEYUP, K.K_o),
                     (4.6, K.KEYDOWN, K.K_ESCAPE)]
            shot_at = 2.6
        elif voice:  # hold V briefly for each canned transcript (see main(): FakeMic / FakeSTT)
            steps = []
            for i in range(6):
                steps += [(0.5 + i * 0.7, K.KEYDOWN, K.K_v), (0.8 + i * 0.7, K.KEYUP, K.K_v)]
            steps.append((5.8, K.KEYDOWN, K.K_ESCAPE))
            shot_at = 3.6
        elif calib:  # J starts, J measures spot 1, J measures spot 2 (same picture: so the two spots can't fit a camera), Esc
            steps = [(0.8, K.KEYDOWN, K.K_j), (1.0, K.KEYUP, K.K_j), (3.5, K.KEYDOWN, K.K_j), (3.7, K.KEYUP, K.K_j),
                     (6.0, K.KEYDOWN, K.K_j), (6.2, K.KEYUP, K.K_j), (9.0, K.KEYDOWN, K.K_ESCAPE)]
            shot_at = 4.5
        elif follow or heel:
            steps = [(0.5, K.KEYDOWN, K.K_h if heel else K.K_t), (0.7, K.KEYUP, K.K_h if heel else K.K_t), (0.9, K.KEYDOWN, K.K_y), (1.1, K.KEYUP, K.K_y),
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
            if listen:
                for i, (when, text) in enumerate(LISTEN_TEST):
                    if t >= when and ("amb", i) not in done:
                        done.add(("amb", i))
                        self.voice_q.put(("ambient", text, -1))
            for i, (when, typ, key) in enumerate(steps):
                if t >= when and i not in done:
                    done.add(i)
                    pygame.event.post(pygame.event.Event(typ, key=key, mod=0, unicode="", scancode=0))
            if t >= shot_at and not state["shot"]:
                state["shot"] = True
                pygame.image.save(screen, "/tmp/go2_selftest_calib.png" if calib else "/tmp/go2_selftest_listen.png" if listen else
                                  "/tmp/go2_selftest_voicemove.png" if voicemove else
                                  "/tmp/go2_selftest_objects.png" if objects else
                                  "/tmp/go2_selftest_voice.png" if voice else
                                  "/tmp/go2_selftest_heel.png" if heel else
                                  "/tmp/go2_selftest_follow.png" if follow else "/tmp/go2_selftest.png")
            if objects and t >= 3.0 and self._obj_snapshot is None and self.objects:
                self._obj_snapshot = list(self.objects[1])   # what the overlay held just before it was switched off
            mark_at = 5.0 if (follow or heel) else 8.0 if voicemove else None
            if mark_at and t >= mark_at and not state["marked"] and isinstance(self.robot, FakeRobot):
                state["marked"] = True  # 0.5 s after Space / 'stop': nothing should move the dog from here on
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


def _selftest_calib_verdict(app: "App") -> int:
    import heel_sim

    ev = app.cal_events
    spots = [e for e in ev if e[0] == "spot"]
    # the last step, driven with measurements from a known camera (0.31 m high, tilted 5 deg): must be recovered and applied
    import random as _random

    rng = _random.Random(2)
    rows = []
    for d in CAL_SPOTS:
        w = heel_sim.World(dog=[0, 0, 0], person=[heel_sim.CAM.x_off + d, 0.0, 0.0], cam_z=0.31, cam_pitch=math.radians(5))
        rows.append((heel_sim.project(w, rng, noise=0)[3], d))
    before = (app.args.heel_cam_height, app.args.heel_cam_pitch)          # what the refused attempt left behind
    ok_fit = app.finish_calibration(rows, 720)
    checks = {
        "J then J: the real detector saw the person and recorded spot 1 (feet row near the bottom of the test picture)": bool(spots) and 400 < spots[0][1] < 700,
        "the pretend lidar located them (about 1.5 m ahead) and calibration went by LIDAR": bool(spots) and ("lidar" in " ".join(app.messages_text())),
        "a second press at the same distance was NOT taken as a new spot (needs a clearly different distance)": ("same_distance",) in ev and len(spots) == 1,
        "...and it changed nothing (height/tilt still the defaults before the good measurements)": before == (0.35, 0.0),
        "measurements from a known camera (0.31 m, 5 deg) are solved and applied": ok_fit and abs(app.args.heel_cam_height - 0.31) < 0.01
                                                                                  and abs(app.args.heel_cam_pitch - 5.0) < 0.5,
        "the result was written to the (temp) calibration file, never the real one": os.path.exists("/tmp/go2_cal_test.json"),
        "Esc left no calibration half-done that could move the dog": not any(e[0] in ("move", "sport") for e in getattr(app.robot, "log", [])),
    }
    print("\nSELFTEST-CALIB events:", ev)
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


def _selftest_heel_verdict(app: "App") -> int:
    log = app.robot.log if isinstance(app.robot, FakeRobot) else []
    sports = [e[1] for e in log if e[0] == "sport"]
    moves = [e for e in log if e[0] == "move"]
    after = log[app._log_len_after_stop:] if app._log_len_after_stop is not None else None
    checks = {
        "balanced first, then e-stop (BalanceStand, StopMove)": sports == ["BalanceStand", "StopMove"],
        "started heeling (H then Y), not plain follow": app.follow_starts == 1 and app.follow_mode == "heel",
        "walked toward the person, who is further ahead than the heel spot (vx > 0.3)": any(m[1] > 0.3 for m in moves),
        "never went faster than --heel-speed": all(m[1] <= app.args.heel_speed + 1e-6 for m in moves),
        "never backed up (the person is ahead of the spot)": all(m[1] >= 0 for m in moves),
        "no walking after Space": after is not None and not any(e[0] == "move" for e in after),
        "heel ended": not app.following,
        "the (pretend) lidar was used: the controller fused it with the camera": "camera+lidar" in app.follow_sources,
    }
    print("\nSELFTEST-HEEL command log:", log)
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
        "'follow me' started following directly (no Y needed)": app.follow_starts == 1,
        "'what is the weather' matched nothing": labels[4] is None,
        "every transcript was handled": labels == ["Hello", "Dance 2", "follow the nearest person", "STOP", None, "stop following"],
        "'stop' ended the follow it had started": not app.following and app.pending is None,
    }
    print("\nSELFTEST-VOICE heard:", app.heard_log)
    print("SELFTEST-VOICE command log:", log)
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


LISTEN_TEST = [  # (seconds, what the always-on mic "heard")
    (0.5, "hello everyone how are you doing today"), (0.9, "sit down"), (1.3, "Ernest sit down"), (2.0, "Ernest"), (2.5, "wave"),
    (3.2, "wave"), (3.6, "Ernest, ready to dance"), (4.2, "Ernest stand on your back legs"), (4.6, "we should get lunch"),
    (5.0, "yes"), (7.2, "Ernest sit down"), (7.8, "Ernest come down"), (9.0, "Ernest box step"), (17.5, "stop"),
]


def _selftest_listen_verdict(app: "App") -> int:
    log = app.robot.log if isinstance(app.robot, FakeRobot) else []
    sports = [e[1] for e in log if e[0] == "sport"]
    api = [e[1:] for e in log if e[0] == "api"]
    acts = [a for _, a in app.ambient_log]
    checks = {
        "conversation and bare commands without the wake word were ignored, in order":
            acts == ["ignore", "ignore", "command", "wake", "command", "ignore", "command", "command", "ignore", "confirm", "command",
                     "command", "command", "stop"],
        "sport commands: Sit, Hello (wake window), StandUp ('ready to dance'), BalanceStand (back-leg prep), "
        "then 'Ernest box step' (no confirmation) = StandUp + BalanceStand, and finally StopMove; nothing lay it down":
            sports == ["Sit", "Hello", "StandUp", "BalanceStand", "StandUp", "BalanceStand", "StopMove"],
        "'box step' stood the dog up, balanced it, and put it on the back legs, in that order":
            log[max(i for i, e in enumerate(log) if e == ("sport", "StandUp")):][:3]
            == [("sport", "StandUp"), ("sport", "BalanceStand"), ("api", 1050, {"data": True})],
        "back legs: 2050 was refused so it fell back to 1050 (up), came down with 1050, and box step went straight to 1050 again":
            api == [(2050, {"data": True}), (1050, {"data": True}), (1050, {"data": False}), (1050, {"data": True})],
        "'sit down' while up on two legs was blocked (Sit only once)": sports.count("Sit") == 1,
        "the dog was never told to sit or lie down (stayed up until 'stop')": "Sit" not in sports[2:] and "StandDown" not in sports,
        "'stop' halted the dog exactly once": sports.count("StopMove") == 1,
        "the box-step session ended on 'stop' (no more dancing); nothing pending; it stays up until 'come down'":
            not app.box_step and not app.box_dancing and app.upright and app.pending is None,
        "box step actually moved the dog (forward/back and side-to-side velocity commands while upright)":
            len({(e[1], e[2]) for e in log if e[0] == "move" and (e[1] or e[2])}) >= 2,
    }
    print("\nSELFTEST-LISTEN ambient:", app.ambient_log)
    print("SELFTEST-LISTEN sports:", sports, " api:", api)
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


VOICEMOVE_LINES = ["walk forward", "turn left", "go back a little", "turn right 45 degrees", "step left", "stop"]


def _selftest_voicemove_verdict(app: "App") -> int:
    log = app.robot.log if isinstance(app.robot, FakeRobot) else []
    moves = [e[1:] for e in log if e[0] == "move"]
    sports = [e[1] for e in log if e[0] == "sport"]
    after = log[app._log_len_after_stop:] if app._log_len_after_stop is not None else None
    L, A = app.args.linear, app.args.angular
    expected = [(round(L, 2), 0.0, 0.0), (0.0, 0.0, round(A, 2)), (round(-L, 2), 0.0, 0.0),
                (0.0, 0.0, round(-A, 2)), (0.0, round(L, 2), 0.0)]
    checks = {
        "each spoken move produced the right velocity, in order (fwd, left, back, right, strafe left)": moves == expected,
        "balanced once before the first move, then 'stop' sent StopMove": sports == ["BalanceStand", "StopMove"],
        "'stop' cancelled the step in progress (no moves afterwards)": after is not None and not any(e[0] == "move" for e in after),
        "no voice move left running": app.voice_move is None,
        "heard all six": [lab for _, lab in app.heard_log] == ["walk forward", "turn left", "walk backward", "turn right", "step left", "STOP"],
    }
    print("\nSELFTEST-VOICEMOVE heard:", app.heard_log)
    print("SELFTEST-VOICEMOVE moves:", moves)
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
    p.add_argument("--selftest-voicemove", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selftest-listen", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--wake-word", default=os.environ.get("GO2_WAKE_WORD", "ernest"),
                   help="always-listening wake word: say it first ('ernest, sit down'); a bare 'stop' works without it")
    p.add_argument("--no-listen", action="store_true", help="don't start the always-on listener (hold V still works)")
    p.add_argument("--upright-api", type=int, default=int(os.environ.get("GO2_UPRIGHT_API", "2050")),
                   help="back-leg stand id tried first: 2050 BackStand (firmware 1.1.7+, the default) or 1050 (older); the other is the fallback")
    p.add_argument("--box-wait", type=float, default=3.2, help="seconds 'box step' waits for the dog to finish standing up before balancing")
    p.add_argument("--linear", type=float, default=LINEAR, help="forward/sideways speed, m/s")
    p.add_argument("--angular", type=float, default=ANGULAR, help="turn speed, rad/s")
    p.add_argument("--follow-speed", type=float, default=0.35, help="max forward speed while following, m/s")
    p.add_argument("--heel-side", choices=["left", "right"], default="left", help="which side of you the dog walks on when heeling")
    p.add_argument("--heel-speed", type=float, default=0.8, help="max forward speed while heeling, m/s (a slow walk is ~1.0; the dog trails you above this)")
    p.add_argument("--heel-lidar", dest="heel_lidar", action="store_true", default=True,
                   help="heel uses the camera AND the dog's lidar (the default): the camera decides who and where you are, the lidar only sharpens "
                        "the distance. UNTESTED on the real dog. If the dog sends no lidar it says so and uses the camera alone")
    p.add_argument("--no-heel-lidar", dest="heel_lidar", action="store_false", help="heel uses the camera only")
    p.add_argument("--heel-lead", type=float, default=1.3, help="heel: how far ahead of the dog's centre you are, m. Smaller = the dog walks closer, but the camera sees less of you (try 0.9)")
    p.add_argument("--heel-gap", type=float, default=0.35, help="heel: sideways distance between you and the dog, m")
    p.add_argument("--heel-cam-height", type=float, default=0.35, help="ESTIMATE: camera height above the floor, m (sets how far away the dog thinks you are)")
    p.add_argument("--heel-cam-pitch", type=float, default=0.0, help="ESTIMATE: degrees the camera looks down")
    p.add_argument("--selftest-heel", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--selftest-calib", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stt", choices=["local", "elevenlabs"], default=os.environ.get("GO2_STT", "local"),
                   help="speech-to-text for the V key: 'local' = offline Whisper (default), 'elevenlabs' = cloud (needs internet + key)")
    p.add_argument("--whisper-model", default=os.environ.get("GO2_WHISPER_MODEL", "small.en"),
                   help="local Whisper model (small.en, the default, copes better with noise; base.en is ~2.4x faster)")
    p.add_argument("--voice-log", default=os.environ.get("GO2_VOICE_LOG"), metavar="DIR",
                   help="save every utterance the always-on mic hears (wav + transcript) into DIR, to study what it gets wrong")
    p.add_argument("--motion-mode", choices=["normal", "ai", "mcf"], default=None,
                   help="optional, UNTESTED: switch the dog's motion controller at connect (DimOS notes 'mcf' is the one that traverses stairs)")
    args = p.parse_args()
    args.cal_loaded = None
    if follow_mod is not None and not any(k.startswith("selftest") and v for k, v in vars(args).items()) \
            and args.heel_cam_height == 0.35 and args.heel_cam_pitch == 0.0:           # (only if nothing was set by hand)
        saved = follow_mod.load_calibration()
        if saved:
            args.heel_cam_height, args.heel_cam_pitch = saved
            args.cal_loaded = saved
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
    if args.selftest_follow or args.selftest_heel or args.selftest_calib:
        image = _follow_test_canvas()
        if image is None:
            return 2
    if args.selftest_objects:
        image = _objects_test_canvas()
        if image is None:
            return 2
    if any(k.startswith("selftest") and v for k, v in vars(args).items()):
        args.no_listen = True                                   # tests never open the mic
        args.heel_lidar = args.heel_lidar or args.selftest_heel  # the heel test also exercises the (pretend) lidar
        args.box_wait = 0.4                                     # (and don't wait 3 s for a fake dog to stand)
    app = App(args, demo_image=image)
    if args.selftest_voice:
        app.mic, app.stt = FakeMic(), FakeSTT(VOICE_TEST_LINES)
    if args.selftest_voicemove:
        app.mic, app.stt = FakeMic(), FakeSTT(VOICEMOVE_LINES)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
