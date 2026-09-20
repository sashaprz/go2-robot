"""The dog app's side of the phone link: reads what the person's iPhone says about them (from phone_server.py, which go2.bat starts) and
turns it into a follow.PhoneState for the follower.

The phone tells us two things that the camera can't when the person is out of its view:
  * walking / standing  (from the accelerometer's rhythm)  -> "they stopped, wait for them" instead of giving up
  * turning rate        (from the gyro; only used if the page's turning set-up was done, and off by default in the controller)
The audio comes separately (voice.PhoneMic).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time

import voice

HERE = os.path.dirname(os.path.abspath(__file__))
MOTION_PORT = 48126


class PhoneLink:
    def __init__(self, host: str | None = None, port: int = MOTION_PORT, token_path: str | None = None):
        self.host = host or voice._windows_host_ip()
        self.port = port
        self.token_path = token_path or os.path.join(HERE, ".phone_wsl_token")
        self.error = ""
        self.connected = False             # to phone_server.py
        self.ua = ""                       # the phone's browser, once it has said hello
        self.yaw_rate = 0.0                # rad/s, + = turning left (as set up on the page)
        self.walking = False
        self.calibrated = False
        self.energy = 0.0
        self.heading = None                # compass heading in degrees, when the phone gives one
        self.last_motion_at = 0.0          # local time of the last motion update
        self.updates = 0
        self._closed = False
        self._thread: threading.Thread | None = None
        self._sock = None

    def start(self) -> None:
        self._closed = False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="phonelink")
            self._thread.start()

    def close(self) -> None:
        self._closed = True
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass

    def apply(self, msg: dict, now: float | None = None) -> None:
        """One JSON message from the phone (split out so it can be tested without a socket)."""
        now = time.time() if now is None else now
        kind = msg.get("type")
        if kind == "hello":
            self.ua = str(msg.get("ua", ""))[:120]
        elif kind == "m":
            try:
                self.yaw_rate = float(msg.get("yaw", 0.0))
                self.walking = bool(msg.get("walk", False))
                self.calibrated = bool(msg.get("cal", False))
                self.energy = float(msg.get("en", 0.0))
                h = msg.get("hdg")
                self.heading = float(h) if h is not None else None
            except (TypeError, ValueError):
                return
            self.last_motion_at = now
            self.updates += 1

    @property
    def age(self) -> float:
        return time.time() - self.last_motion_at if self.last_motion_at else 99.0

    @property
    def live(self) -> bool:
        """The phone is sending its motion right now."""
        return self.age < 1.5

    def state(self):
        """A follow.PhoneState, or None when the phone isn't sending (then the controller just uses the camera)."""
        from follow import PhoneState

        if not self.live:
            return None
        return PhoneState(yaw_rate=self.yaw_rate, walking=self.walking, calibrated=self.calibrated, age=self.age)

    def _run(self) -> None:
        while not self._closed:
            try:
                with open(self.token_path, encoding="ascii") as f:
                    token = f.read().strip()
            except OSError:
                self.error = "the phone server hasn't started (no .phone_wsl_token): launch go2.bat with the phone link on"
                time.sleep(1.0)
                continue
            sock = socket.socket()
            sock.settimeout(4.0)
            try:
                sock.connect((self.host, self.port))
                sock.sendall(token.encode("ascii"))
            except OSError as e:
                self.error = f"can't reach the phone server at {self.host}:{self.port} ({type(e).__name__})"
                sock.close()
                time.sleep(1.0)
                continue
            self._sock, self.connected, self.error = sock, True, ""
            sock.settimeout(2.0)
            buf = b""
            try:
                while not self._closed:
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue                                # a quiet phone isn't an error; `live` handles it
                    if not data:
                        self.error = "the phone server closed the connection"
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            self.apply(json.loads(line.decode("utf-8", "replace")))
                        except ValueError:
                            pass
                    if len(buf) > 65536:
                        buf = b""
            except OSError as e:
                if not self._closed:
                    self.error = f"lost the phone server ({type(e).__name__})"
            finally:
                self.connected = False
                sock.close()
            if not self._closed:
                time.sleep(1.0)
