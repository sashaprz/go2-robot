"""Read-only check of what the dog's lidar delivers. Run it with the dog on and its Wi-Fi joined (lidar-probe.bat).

It turns the lidar on, listens for ~20 s and prints, once a second: how many lidar messages / pose messages arrived,
how many points, and the nearest obstacle in each direction around the dog (from obstacles.py). It NEVER moves the dog.

What to try while it runs (tells us whether "sit when something enters the path" is workable):
  1. Stand still: do the numbers settle, and does "front" match a wall/box you can measure with a tape?
  2. Step into the space in front of the dog: how many seconds before "front" drops? (>1 s would be too slow to stop for.)
  3. Walk away again: how long until it reads clear?
  4. Stand at the dog's side (~0.6 m): does "left"/"right" show you? (That is how heel could know you're beside it.)

  lidar-probe.bat 60 save     also saves what the lidar saw (points near the dog + its pose) to lidar_recording.npz, so a
                              person tracker can be built and tested on REAL data. Suggested 60 s: 10 s standing beside the
                              dog (0.6 m), 15 s walking slowly beside it, 10 s standing near a wall, 10 s with nobody near.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

import obstacles

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "save" else 20.0
SAVE = "save" in sys.argv[1:]
STAGES = [(0, "STAND still 0.6 m beside the dog (on its side)"), (10, "WALK slowly beside the dog"),
          (25, "STAND still near a wall, with the dog next to you"), (35, "MOVE AWAY: nobody near the dog"), (50, "stay away, nearly done")]
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_recording.npz")


def main() -> int:
    key = os.environ.get("UNITREE_AES_128_KEY")
    if not key:
        print("UNITREE_AES_128_KEY isn't set (expected in ~/.dimos.env)")
        return 2
    import asyncio

    from unitree_webrtc_connect.constants import RTC_TOPIC

    from dimos.robot.unitree.connection import UnitreeWebRTCConnection

    ip = os.environ.get("ROBOT_IP", "192.168.12.1")
    print(f"connecting to {ip} ...", flush=True)
    c = UnitreeWebRTCConnection(ip, aes_128_key=key)
    lock = threading.Lock()
    state = {"lidar": 0, "pose": 0, "pts": None, "pose_v": None, "lidar_t": [], "keys": None, "rec": []}

    def on_lidar(msg):
        try:
            d = msg["data"]
            pts = np.asarray(d["data"]["points"], dtype=np.float32).reshape(-1, 3)
        except Exception as e:  # noqa: BLE001 - report the shape we got instead of guessing
            with lock:
                state["keys"] = f"unexpected lidar message ({type(e).__name__}: {e}); top-level keys: {list(msg)[:8]}"
            return
        with lock:
            state["lidar"] += 1
            state["pts"] = pts
            state["lidar_t"].append(time.time())
            if SAVE and state["pose_v"] is not None:          # points within 4 m, in the dog's frame (small and precise)
                dog = obstacles.to_dog_frame(pts, *state["pose_v"])
                near = dog[(np.abs(dog[:, 0]) < 4) & (np.abs(dog[:, 1]) < 4) & (dog[:, 2] < 2.2)]
                state["rec"].append((time.time(), state["pose_v"], near.astype(np.float16)))
            if state["keys"] is None:
                state["keys"] = f"data keys: {list(d)[:8]}, inner keys: {list(d['data'])[:8]}"

    def on_pose(p):
        with lock:
            state["pose"] += 1
            state["pose_v"] = (p.position.x, p.position.y, p.position.z, p.yaw)

    subs = [c.raw_lidar_stream().subscribe(on_lidar), c.odom_stream().subscribe(on_pose)]
    # the lidar only publishes once switched on (DimOS doesn't do this itself)
    asyncio.run_coroutine_threadsafe(
        _switch_on(c, RTC_TOPIC["ULIDAR_SWITCH"]), c.loop).result(timeout=8)
    print(f"listening for {SECONDS:.0f} s (never moves the dog). Try the four things in the header of this file.\n", flush=True)
    t0 = time.time()
    shown = set()
    try:
        while time.time() - t0 < SECONDS:
            time.sleep(1.0)
            if SAVE and SECONDS >= 55:                           # tell the person what to do, and when
                for at, what in STAGES:
                    if time.time() - t0 >= at and at not in shown:
                        shown.add(at)
                        print(f"\n>>> NOW ({at}-s mark): {what}\n", flush=True)
            with lock:
                n_l, n_p, pts, pv, ts, keys = (state["lidar"], state["pose"], state["pts"], state["pose_v"],
                                               list(state["lidar_t"]), state["keys"])
            recent = [t for t in ts if t > time.time() - 3]
            rate = len(recent) / 3 if recent else 0.0
            line = f"t={time.time() - t0:4.0f}s  lidar msgs {n_l} ({rate:.1f}/s)  pose msgs {n_p}"
            if pts is not None and pv is not None:
                dog = obstacles.to_dog_frame(pts, *pv)
                r = obstacles.sector_ranges(dog)
                clr = obstacles.clearance(dog)
                line += (f"  {len(pts)} pts, z {pts[:, 2].min():.2f}..{pts[:, 2].max():.2f}"
                         f"  nearest: front {r['front']:.2f} left {r['left']:.2f} right {r['right']:.2f} back {r['back']:.2f} m"
                         f"  corridor: {'clear' if clr is None else f'{clr:.2f} m'}")
            print(line, flush=True)
        print()
        if SAVE:
            rec = state["rec"]
            if rec:
                np.savez_compressed(SAVE_PATH, t=np.array([r[0] for r in rec]), pose=np.array([r[1] for r in rec]),
                                    counts=np.array([len(r[2]) for r in rec]), pts=np.concatenate([r[2] for r in rec]))
                print(f"saved {len(rec)} lidar frames to {SAVE_PATH}")
            else:
                print("nothing to save: no lidar frames arrived")
        if keys:
            print(keys)
        if state["lidar"] == 0:
            print("RESULT: no lidar messages arrived. This dog isn't sending lidar over this connection (or it needs the "
                  "switch topic in a different form): guide mode would have to use the camera only.")
        elif state["pose"] == 0:
            print("RESULT: lidar arrived but no pose messages: obstacle positions can't be placed relative to the dog.")
        else:
            print("RESULT: lidar and pose both arrive. Check the numbers above against a tape measure before trusting them.")
    finally:
        for s in subs:
            try:
                s.dispose()
            except Exception:  # noqa: BLE001
                pass
        try:
            c.stop()
        except Exception:  # noqa: BLE001
            pass
    return 0


async def _switch_on(c, topic: str) -> None:
    c.conn.datachannel.pub_sub.publish_without_callback(topic, "on")


if __name__ == "__main__":
    raise SystemExit(main())
