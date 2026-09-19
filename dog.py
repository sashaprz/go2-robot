"""Keyboard controller for a Unitree Go2 Air over its Wi-Fi hotspot (WebRTC, LocalAP).

Run via dog.bat AFTER joining the Go2_61331_29d4be72 Wi-Fi network. Needs no internet.
"""
import asyncio
import json
import logging
import msvcrt
import subprocess
import sys
import time

from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

DOG_IP = "192.168.12.1"
DOG_SSID = "Go2_61331_29d4be72"
DOG_WIFI_PASSWORD = "88888888"

# Deliberately slow. Raise these once you're comfortable.
VX, VY, VYAW = 0.3, 0.25, 0.6      # m/s forward, m/s sideways, rad/s turn
MOVE_HOLD_S = 0.55                 # a keypress keeps moving this long; release = stops (dead-man)
MOVE_HZ = 10

MOVE_KEYS = {                      # key -> (x, y, yaw)
    "w": (VX, 0, 0), "s": (-VX, 0, 0),
    "a": (0, VY, 0), "d": (0, -VY, 0),
    "q": (0, 0, VYAW), "e": (0, 0, -VYAW),
}
ACTION_KEYS = {                    # key -> (label, sport command)
    "1": ("Stand up", "StandUp"),
    "2": ("Lie down", "StandDown"),
    "3": ("Recovery stand (after a fall)", "RecoveryStand"),
    "4": ("Sit", "Sit"),
    "5": ("Rise from sit", "RiseSit"),
    "h": ("Hello", "Hello"),
    "t": ("Stretch", "Stretch"),
    "c": ("Content", "Content"),
    "g": ("Wiggle hips", "WiggleHips"),
}

HELP = """
  MOVE (tap or hold; releasing stops the dog)     POSES / TRICKS
    w / s   forward / back                          1 stand up      2 lie down
    a / d   strafe left / right                     3 recovery stand (if it fell)
    q / e   turn left / right                       4 sit           5 rise from sit
    SPACE   stop immediately                        h hello   t stretch
                                                    c content g wiggle hips
  b  battery      ?  this help      x  quit (stops the dog first)

  Keep the area around the dog clear. Slow speeds are set at the top of dog.py.
"""

logging.basicConfig(level=logging.ERROR)


def dog_reachable() -> bool:
    r = subprocess.run(["ping", "-n", "1", "-w", "1500", DOG_IP], capture_output=True, text=True)
    return "TTL=" in r.stdout


def wait_for_dog(timeout_s: int = 30) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if dog_reachable():
            return True
        time.sleep(1)
    return False


class Dog:
    def __init__(self):
        self.conn = None
        self.battery = None
        self.move_until = 0.0
        self.move_vec = (0.0, 0.0, 0.0)
        self.moving = False

    @property
    def ps(self):
        return self.conn.datachannel.pub_sub

    async def connect(self):
        last_err = None
        for attempt in range(1, 4):
            try:
                self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
                await self.conn.connect()
                break
            except Exception as e:  # noqa: BLE001 - retry on anything, report the last one
                last_err = e
                print(f"  connect attempt {attempt}/3 failed: {e}")
                await asyncio.sleep(2)
        else:
            raise RuntimeError(f"could not connect: {last_err}")

        self.ps.subscribe(RTC_TOPIC["LOW_STATE"], self._on_low_state)
        await self._ensure_normal_mode()

    def _on_low_state(self, message):
        soc = message.get("data", {}).get("bms_state", {}).get("soc")
        if soc is not None:
            self.battery = soc

    async def _ensure_normal_mode(self):
        """Sport commands only work in 'normal' motion mode."""
        try:
            resp = await asyncio.wait_for(
                self.ps.publish_request_new(RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}), 5)
            mode = json.loads(resp["data"]["data"]).get("name")
            if mode != "normal":
                print(f"  switching motion mode {mode!r} -> 'normal'")
                await asyncio.wait_for(self.ps.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"],
                    {"api_id": 1002, "parameter": {"name": "normal"}}), 5)
                await asyncio.sleep(4)
        except Exception as e:  # noqa: BLE001
            print(f"  (couldn't check motion mode: {e}; continuing)")

    async def sport(self, name: str):
        await self.ps.publish_request_new(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD[name]})

    def _send_move(self, x, y, yaw):
        task = asyncio.ensure_future(self.ps.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": yaw}}))
        task.add_done_callback(lambda t: t.exception())  # fire-and-forget; ignore late errors

    async def stop(self):
        self.move_until = 0.0
        self.moving = False
        try:
            await self.sport("StopMove")
        except Exception:  # noqa: BLE001
            pass

    async def tick(self):
        """Called every 1/MOVE_HZ s: keep moving while a key is 'held', stop when it lapses."""
        if time.time() < self.move_until:
            self._send_move(*self.move_vec)
            self.moving = True
        elif self.moving:
            await self.stop()

    async def close(self):
        await self.stop()
        try:
            await self.conn.disconnect()
        except Exception:  # noqa: BLE001
            pass


async def main():
    print(f"Go2 controller. Looking for the dog at {DOG_IP} ...")
    if not await asyncio.to_thread(wait_for_dog, 30):
        print(f"""
Can't reach the dog. Make sure the PC is on the dog's Wi-Fi:
    network:  {DOG_SSID}
    password: {DOG_WIFI_PASSWORD}
and that the dog is powered on. Then run this again.
(Windows may drop the network because it has no internet - if so, re-select it.)""")
        return 1

    dog = Dog()
    print("Dog found. Connecting (takes a few seconds) ...")
    try:
        await dog.connect()
    except Exception as e:  # noqa: BLE001
        print(f"\nConnection failed: {e}")
        print("Tips: close the Unitree Go phone app (only one controller can connect at a time), "
              "then power-cycle the dog if it still fails.")
        return 1

    print("\nConnected!" + HELP)
    try:
        while True:
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):      # arrow/function keys send a 2-char sequence
                    msvcrt.getwch()
                    continue
                k = ch.lower()
                if k in MOVE_KEYS:
                    dog.move_vec = MOVE_KEYS[k]
                    dog.move_until = time.time() + MOVE_HOLD_S
                elif k == " ":
                    await dog.stop()
                    print("  stop")
                elif k in ACTION_KEYS:
                    label, cmd = ACTION_KEYS[k]
                    await dog.stop()
                    print(f"  {label}")
                    await dog.sport(cmd)
                elif k == "b":
                    print(f"  battery: {dog.battery}%" if dog.battery is not None
                          else "  battery: (no reading yet)")
                elif k == "?":
                    print(HELP)
                elif k == "x" or ch == "\x03":
                    return 0
            await dog.tick()
            await asyncio.sleep(1 / MOVE_HZ)
    finally:
        print("Stopping and disconnecting ...")
        await dog.close()


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        code = 0
    sys.exit(code)
