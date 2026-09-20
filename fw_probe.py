"""Read-only firmware check. Run it with the dog on and its Wi-Fi joined (fw-probe.bat). NEVER moves the dog.

Connects, listens ~8 s on the dog's status topics, and prints any field whose name or value looks like a version
(version / firmware / sw / hw / build), plus the motion-mode reply. If nothing turns up, it says so; the version is then
only in the Unitree Go app (Device -> Settings).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
KEY_RE = re.compile(r"version|firmware|fw|sw_|hw_|build|sn$|serial", re.I)
VAL_RE = re.compile(r"^\d+\.\d+\.\d+")

TOPICS = ["SERVICE_STATE", "SELF_TEST", "MULTIPLE_STATE", "LOW_STATE", "SPORT_MOD_STATE", "UWB_STATE",
          "GPT_FEEDBACK", "AUDIO_HUB_PLAY_STATE"]


def walk(obj, path=""):
    """Yield (path, value) for every scalar in a nested message."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:50]):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def main() -> int:
    key = os.environ.get("UNITREE_AES_128_KEY")
    if not key:
        print("UNITREE_AES_128_KEY isn't set (expected in ~/.dimos.env)")
        return 2
    from unitree_webrtc_connect.constants import RTC_TOPIC

    from dimos.robot.unitree.connection import UnitreeWebRTCConnection

    ip = os.environ.get("ROBOT_IP", "192.168.12.1")
    print(f"connecting to {ip} ...", flush=True)
    c = UnitreeWebRTCConnection(ip, aes_128_key=key)
    lock = threading.Lock()
    seen: dict[str, int] = {}
    hits: dict[str, object] = {}

    def make_cb(name):
        def cb(msg):
            try:
                data = msg.get("data", msg) if isinstance(msg, dict) else msg
                with lock:
                    seen[name] = seen.get(name, 0) + 1
                    for p, v in walk(data):
                        if KEY_RE.search(p.split(".")[-1]) or (isinstance(v, str) and VAL_RE.match(v)):
                            hits[f"{name}:{p}"] = v
            except Exception as e:  # noqa: BLE001 - never let a odd message kill the probe
                with lock:
                    hits[f"{name}:<parse error>"] = f"{type(e).__name__}: {e}"
        return cb

    ps = c.conn.datachannel.pub_sub

    def subscribe_all():
        for t in TOPICS:
            ps.subscribe(RTC_TOPIC[t], make_cb(t))

    c.loop.call_soon_threadsafe(subscribe_all)

    try:
        coro = ps.publish_request_new(RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001})
        resp = asyncio.run_coroutine_threadsafe(coro, c.loop).result(timeout=8)
        print("motion switcher:", json.loads(resp["data"]["data"]))
    except Exception as e:  # noqa: BLE001
        print(f"motion switcher query failed: {type(e).__name__}: {e}")

    print(f"listening {SECONDS:.0f} s (never moves the dog) ...\n", flush=True)
    time.sleep(SECONDS)
    try:
        with lock:
            print("messages per topic:", seen or "none")
            if hits:
                print("\nversion-like fields:")
                for k, v in sorted(hits.items()):
                    print(f"  {k} = {v!r}")
            else:
                print("\nno version-like fields in any status message.")
    finally:
        try:
            c.stop()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
