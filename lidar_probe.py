"""Read-only check of what the dog's lidar delivers. Run it with the dog on and its Wi-Fi joined (lidar-probe.bat).

It turns the lidar on, listens (60 s by default) and prints, once a second: how many lidar messages / pose messages arrived,
how many points, and the nearest obstacle in each direction around the dog (from obstacles.py). It NEVER moves the dog.

It ALWAYS leaves two files next to this script, even if it fails or you close it early:
  lidar_probe_report.txt   everything printed, plus the result and any errors  (rewritten every second)
  lidar_recording.npz      the lidar frames (points within 4 m of the dog + its pose), when `save` is given (default in the .bat)

Suggested 60 s: 0-10 s stand still 0.6 m beside the dog, 10-25 s walk slowly beside it, 25-35 s stand still near a wall,
35-60 s nobody near the dog. The window prints the steps.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback

import numpy as np

import obstacles

HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "save" else 20.0
SAVE = "save" in sys.argv[1:]
SAVE_PATH = os.path.join(HERE, "lidar_recording.npz")
REPORT_PATH = os.path.join(HERE, "lidar_probe_report.txt")
STAGES = [(0, "STAND still 0.6 m beside the dog (on its side)"), (10, "WALK slowly beside the dog"),
          (25, "STAND still near a wall, with the dog next to you"), (35, "MOVE AWAY: nobody near the dog"), (50, "stay away, nearly done")]

REPORT: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    REPORT.append(text)
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT) + "\n")
    except OSError:
        pass


def main() -> int:
    key = os.environ.get("UNITREE_AES_128_KEY")
    if not key:
        say("UNITREE_AES_128_KEY isn't set (expected in ~/.dimos.env)")
        return 2
    import asyncio

    from unitree_webrtc_connect.constants import RTC_TOPIC

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
                say("  The dog accepted the network but the connection never completed: something else is probably connected to it.\n"
                    "  Close the Unitree Go phone app COMPLETELY (swipe it away), and any go2.bat / other dog window. Waiting 20 s, then retrying ...")
                time.sleep(20)
    if c is None:
        say("\nCOULD NOT CONNECT to the dog after 3 tries. This is a connection problem, NOT a lidar result.\n"
            "  1. Force-close the Unitree Go phone app (and turn the phone's Wi-Fi off for now).\n"
            "  2. Close every other window that talks to the dog (go2.bat, corridor.bat, fw-probe.bat, dimos-go2.bat).\n"
            "  3. If it still fails: power the dog off and on (that clears a stuck connection), rejoin its Wi-Fi, and run go2.bat first: "
            "if go2.bat connects, this will too.")
        return 3
    say("connected")
    lock = threading.Lock()
    state = {"lidar": 0, "pose": 0, "pts": None, "pose_v": None, "lidar_t": [], "keys": None, "rec": [], "errors": [], "first_pts": None}

    def note_error(where: str) -> None:
        with lock:
            if len(state["errors"]) < 5:
                state["errors"].append(f"{where}: {traceback.format_exc(limit=3)}")

    def on_lidar(msg):
        try:
            d = msg["data"]
            pts = np.asarray(d["data"]["points"], dtype=np.float32).reshape(-1, 3)
        except Exception:  # noqa: BLE001 - report the shape we got instead of guessing
            with lock:
                state["keys"] = f"unexpected lidar message shape; top-level keys: {list(msg)[:8] if hasattr(msg, '__iter__') else type(msg)}"
            note_error("lidar message")
            return
        try:
            with lock:
                state["lidar"] += 1
                state["pts"] = pts
                state["lidar_t"].append(time.time())
                if state["first_pts"] is None:
                    state["first_pts"] = (len(pts), pts.min(axis=0).round(2).tolist(), pts.max(axis=0).round(2).tolist())
                if state["keys"] is None:
                    state["keys"] = f"data keys: {list(d)[:8]}, inner keys: {list(d['data'])[:8]}"
                pv = state["pose_v"]
            if SAVE and pv is not None:                          # points within 4 m, in the dog's frame (small and precise)
                dog = obstacles.to_dog_frame(pts, *pv)
                near = dog[(np.abs(dog[:, 0]) < 4) & (np.abs(dog[:, 1]) < 4) & (dog[:, 2] < 2.2)]
                with lock:
                    state["rec"].append((time.time(), pv, near.astype(np.float16)))
        except Exception:  # noqa: BLE001
            note_error("recording a lidar frame")

    def on_pose(p):
        try:
            with lock:
                state["pose"] += 1
                state["pose_v"] = (p.position.x, p.position.y, p.position.z, p.yaw)
        except Exception:  # noqa: BLE001
            note_error("pose message")

    def finish() -> None:
        with lock:
            rec, n_l, n_p, keys, first, errs = state["rec"], state["lidar"], state["pose"], state["keys"], state["first_pts"], list(state["errors"])
        say()
        if SAVE:
            if rec:
                np.savez_compressed(SAVE_PATH, t=np.array([r[0] for r in rec]), pose=np.array([r[1] for r in rec]),
                                    counts=np.array([len(r[2]) for r in rec]), pts=np.concatenate([r[2] for r in rec]))
                say(f"saved {len(rec)} lidar frames to {SAVE_PATH}")
            else:
                say("nothing to save: no lidar frame arrived together with a pose")
        if keys:
            say(keys)
        if first:
            say(f"first lidar message: {first[0]} points, min xyz {first[1]}, max xyz {first[2]}")
        for e in errs:
            say("ERROR " + e)
        if n_l == 0:
            say("RESULT: no lidar messages arrived. This dog isn't sending lidar over this connection (or it needs the "
                "switch topic in a different form): guide mode would have to use the camera only.")
        elif n_p == 0:
            say("RESULT: lidar arrived but no pose messages: obstacle positions can't be placed relative to the dog.")
        else:
            say(f"RESULT: lidar ({n_l} msgs) and pose ({n_p} msgs) both arrive. Check the numbers above against a tape measure before trusting them.")

    subs = []
    try:
        subs = [c.raw_lidar_stream().subscribe(on_lidar), c.odom_stream().subscribe(on_pose)]
        # the lidar only publishes once switched on (DimOS doesn't do this itself)
        asyncio.run_coroutine_threadsafe(_switch_on(c, RTC_TOPIC["ULIDAR_SWITCH"]), c.loop).result(timeout=8)
        say(f"listening for {SECONDS:.0f} s (never moves the dog)\n")
        t0, shown = time.time(), set()
        while time.time() - t0 < SECONDS:
            time.sleep(1.0)
            if SAVE and SECONDS >= 55:                           # tell the person what to do, and when
                for at, what in STAGES:
                    if time.time() - t0 >= at and at not in shown:
                        shown.add(at)
                        say(f"\n>>> NOW ({at}-s mark): {what}\n")
            with lock:
                n_l, n_p, pts, pv, ts = state["lidar"], state["pose"], state["pts"], state["pose_v"], list(state["lidar_t"])
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
            say(line)
    except BaseException as e:  # noqa: BLE001 - Ctrl-C, a dropped connection, anything: still write what we have
        say(f"STOPPED EARLY: {type(e).__name__}: {e}")
        say(traceback.format_exc(limit=4))
    finally:
        finish()
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
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - e.g. can't reach the dog at all: leave the reason in the report file
        say("PROBE FAILED BEFORE LISTENING:\n" + traceback.format_exc(limit=6))
        raise SystemExit(1)
