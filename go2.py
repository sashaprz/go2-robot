#!/usr/bin/env python3
"""Go2 all-in-one: live camera (FPV) + keyboard driving + poses/tricks in ONE window.

Run via go2.bat (after joining the dog's Wi-Fi). Needs no internet. The window must have keyboard focus.
Reuses DimOS's own connection class, so the per-device AES key works exactly like dimos-go2.bat.
The dog accepts ONE controller at a time: don't run this alongside dimos-go2.bat / the phone app.

  python go2.py            connect to the real dog
  python go2.py --demo     same window with a FAKE robot and synthetic video (no dog needed)
  python go2.py --selftest headless scripted test of the key logic (used during development)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

if "--selftest" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
elif sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")  # WSLg; same driver DimOS's keyboard window forces
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

# ---- tunables ---------------------------------------------------------------------------------------------
LINEAR = 0.4      # m/s forward / sideways
ANGULAR = 0.8     # rad/s turning
BOOST = 1.5       # Shift multiplier
SLOW = 0.5        # Ctrl multiplier
CONTROL_HZ = 20   # velocity command rate (the connection also self-stops 0.2 s after the last command)
CONFIRM_SECS = 4  # how long a dance/routine waits for the confirm key

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
# EDIT FREELY. Timed lists of the sport commands above; each step waits that command's busy time.
ROUTINES = {"greeting": ["StandUp", "BalanceStand", "Hello", "Content", "WiggleHips", "Sit"]}
# Deliberately absent: flips, handstand, bound. They can hurt the robot and most aren't supported on an Air.

WIN_W, WIN_H = 1024, 680
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
    """No dog: records commands and streams synthetic video (for --demo and --selftest)."""

    def __init__(self):
        self.log: list[tuple] = []
        self._halt = threading.Event()

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


# ---- the app ----------------------------------------------------------------------------------------------
class App:
    def __init__(self, args):
        self.args = args
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
        self.frame_times: deque = deque(maxlen=40)
        self.last_frame_at = 0.0
        self.battery = None
        self._fonts: dict = {}
        self._scaled = None
        self.running = True

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
            if self.args.demo or self.args.selftest:
                r = FakeRobot()
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

    # -- actions ----------------------------------------------------------------------------------------
    def run_trick(self, label: str, name: str, wait: float) -> None:
        self.say(f"> {label}")
        self.busy_until = time.time() + wait
        self.armed = name == "BalanceStand"  # after any other pose/trick, re-balance before driving
        self.pool.submit(self.send, name)

    def run_routine(self, rname: str) -> None:
        self.abort.clear()

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
        if key in TRICKS:
            self.run_trick(*TRICKS[key])
        elif key in CONFIRM_TRICKS:
            t = CONFIRM_TRICKS[key]
            self.pending = (t[0], lambda t=t: self.run_trick(*t), now + CONFIRM_SECS)
        elif key == ROUTINE_KEY:
            rname = next(iter(ROUTINES))
            self.pending = (f"routine '{rname}'", lambda: self.run_routine(rname), now + CONFIRM_SECS)

    def update_velocity(self) -> None:
        now, h = time.time(), self.held
        fwd = (pygame.K_w in h) - (pygame.K_s in h)
        side = (pygame.K_q in h) - (pygame.K_e in h)      # Q = left, E = right
        turn = (pygame.K_a in h) - (pygame.K_d in h)      # A = turn left, D = turn right
        if not (fwd or side or turn) or self.robot is None:
            self.desired = (0.0, 0.0, 0.0)
            return
        if now < self.busy_until:
            self.desired = (0.0, 0.0, 0.0)
            return
        if not self.armed:  # wireless-controller velocity only works in BalanceStand
            self.armed = True
            self.busy_until = now + 1.0
            self.say("> balancing before driving ...")
            self.pool.submit(self.send, "BalanceStand")
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
        if self._scaled is not None:
            screen.blit(self._scaled, ((WIN_W - self._scaled.get_width()) // 2, (VIDEO_H - self._scaled.get_height()) // 2))
        else:
            self.text(screen, "waiting for video ...", WIN_W // 2 - 110, VIDEO_H // 2, DIM, 32)
        if self.last_frame_at and time.time() - self.last_frame_at > 3:
            self.text(screen, "NO VIDEO", WIN_W // 2 - 70, VIDEO_H // 2 - 20, BAD, 44)

        now = time.time()
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

        if self.pending and now < self.pending[2]:
            self.text(screen, f"{self.pending[0]}: press Y to confirm ({self.pending[2] - now:.0f}s)", 12, 44, WARN, 34)
        recent = [(msg, color) for t, msg, color in self.messages if now - t < 8]
        if recent:
            y = VIDEO_H - 26 * len(recent) - 10
            backing = pygame.Surface((520, 26 * len(recent) + 8), pygame.SRCALPHA)
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
            "Click this window so it has keyboard focus.   Esc quits.",
        ]
        for i, line in enumerate(lines):
            self.text(screen, line, 12, VIDEO_H + 10 + i * 24, TXT if i < 3 else DIM, 21)

    # -- main loop --------------------------------------------------------------------------------------
    def run(self) -> int:
        pygame.init()
        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SWSURFACE)
        pygame.display.set_caption("Go2 - FPV / drive / tricks")
        threading.Thread(target=self.connect_bg, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        clock = pygame.time.Clock()
        script = self.selftest_script() if self.args.selftest else None
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
                    elif ev.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
                        self.held.clear()  # KEYUPs never arrive once focus is lost: don't keep walking
                        self.say("window lost focus: stopped", WARN)
                self.update_velocity()
                self.draw(screen)
                pygame.display.flip()
                clock.tick(30)
        finally:
            print("Stopping and disconnecting ...", flush=True)
            self.stop_evt.set()
            self.abort.set()
            self.desired = (0.0, 0.0, 0.0)
            if self.robot is not None:
                try:
                    self.robot.stop_move()
                except Exception:  # noqa: BLE001
                    pass
                self.robot.close()
            pygame.quit()
        return self.selftest_verdict() if self.args.selftest else 0

    # -- headless self-test (development only) ----------------------------------------------------------
    def selftest_script(self):
        K = pygame
        steps = [  # (seconds, event type, key)
            (0.6, K.KEYDOWN, K.K_w), (2.2, K.KEYUP, K.K_w),          # arms, then drives forward
            (2.5, K.KEYDOWN, K.K_n), (2.7, K.KEYDOWN, K.K_w), (2.9, K.KEYUP, K.K_w),  # dance pending, cancelled by W
            (3.1, K.KEYDOWN, K.K_n), (3.3, K.KEYDOWN, K.K_y),        # dance confirmed
            (3.5, K.KEYDOWN, K.K_q), (3.7, K.KEYUP, K.K_q),          # driving locked out while dancing
            (3.9, K.KEYDOWN, K.K_SPACE),                              # e-stop clears the lockout
            (4.1, K.KEYDOWN, K.K_a), (5.6, K.KEYUP, K.K_a),          # re-arms, turns left
            (6.2, K.KEYDOWN, K.K_ESCAPE),
        ]
        done = set()
        shot = {"taken": False}

        def script(t, screen):
            for i, (when, typ, key) in enumerate(steps):
                if t >= when and i not in done:
                    done.add(i)
                    pygame.event.post(pygame.event.Event(typ, key=key, mod=0, unicode="", scancode=0))
            if t >= 5.0 and not shot["taken"]:
                shot["taken"] = True
                pygame.image.save(screen, "/tmp/go2_selftest.png")

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default=os.environ.get("ROBOT_IP", "192.168.12.1"))
    p.add_argument("--demo", action="store_true", help="fake robot + synthetic video, no dog needed")
    p.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--linear", type=float, default=LINEAR, help="forward/sideways speed, m/s")
    p.add_argument("--angular", type=float, default=ANGULAR, help="turn speed, rad/s")
    p.add_argument("--motion-mode", choices=["normal", "ai", "mcf"], default=None,
                   help="optional, UNTESTED: switch the dog's motion controller at connect (DimOS notes 'mcf' is the one that traverses stairs)")
    return App(p.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
