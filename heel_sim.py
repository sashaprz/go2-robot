"""Simulated walks for the heel controller in follow.py: no dog, no camera, no detector model needed.

A person walks a scripted path; a pinhole model of the Go2's front camera (real intrinsics, 1280x720) turns their
position into the box a person detector would return (with pixel noise and dropped frames); follow.Heeler steers a
simulated dog that lags its commands like a real one. Prints how well the dog held its spot.

  python heel_sim.py             all scenarios
  python heel_sim.py -v          also print a time line for each
This checks the CONTROL LOGIC only. The camera height / pitch used to turn a box into a distance are estimates, so one
scenario deliberately gives the sim camera different values from the ones the controller assumes.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass

import numpy as np

import follow

CAM = follow.Camera()
PERSON_H, PERSON_W = 1.7, 0.5          # m
LATENCY = 0.15                         # s from a frame to the dog acting on it (detector + network + gait)
TAU = 0.25                             # s, first-order lag of the dog's velocity
DET_HZ = 15.0
LIDAR_EVERY = 2                         # a lidar cloud on every 2nd detection tick (~8 Hz)


@dataclass
class World:
    dog: list                          # x, y, heading  (world frame)
    person: list                       # x, y, heading
    cam_z: float = CAM.z               # the SIM camera's true height / pitch (the controller assumes CAM's)
    cam_pitch: float = CAM.pitch


def to_dog_frame(w: World) -> tuple[float, float]:
    dx, dy = w.person[0] - w.dog[0], w.person[1] - w.dog[1]
    c, s = math.cos(w.dog[2]), math.sin(w.dog[2])
    return dx * c + dy * s, -dx * s + dy * c            # (ahead, left)


def project(w: World, rng: random.Random, noise: float = 6.0):
    """The detector's box for the person, or None when they're out of view (partly-visible people still count)."""
    ahead, left = to_dog_frame(w)
    F, L = ahead - CAM.x_off, left                      # from the lens: forward, left
    sp, cp = math.sin(w.cam_pitch), math.cos(w.cam_pitch)

    def pix(fwd, lat, down):
        Z, Y = fwd * cp + down * sp, -fwd * sp + down * cp
        if Z < 0.15:
            return None
        return CAM.cx + CAM.fx * (-lat) / Z, CAM.cy + CAM.fy * Y / Z, Z

    feet, head = pix(F, L, w.cam_z), pix(F, L, w.cam_z - PERSON_H)
    if feet is None or head is None:
        return None
    hw = CAM.fx * PERSON_W / 2 / feet[2]
    x1, x2, y1, y2 = feet[0] - hw, feet[0] + hw, head[1], feet[1]
    cx1, cx2, cy1, cy2 = max(x1, 0), min(x2, CAM.width), max(y1, 0), min(y2, CAM.height)
    if cx2 - cx1 < 0.5 * (x2 - x1) or cy2 <= cy1 or (cy2 - cy1) / CAM.height < follow.MIN_BOX_H:
        return None
    n = lambda: rng.gauss(0, noise)  # noqa: E731
    return (max(cx1 + n(), 0), max(cy1 + n(), 0), min(cx2 + n(), CAM.width), min(cy2 + n() if cy2 < CAM.height else cy2, CAM.height), 0.9)


def lidar_cloud(w: World, rng: random.Random, wall_y: float | None = None) -> np.ndarray:
    """What the lidar returns, in the dog's frame (x ahead, y left, z up from the floor): the person as a tall blob of
    points (all round: no field of view), and optionally a long wall at world y = wall_y as a distractor."""
    ahead, left = to_dog_frame(w)
    pts = []
    if 0.3 < math.hypot(ahead, left) < 5.0 and rng.random() > 0.1:                 # 10% of scans miss them
        for _ in range(40):
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.1, 0.22)
            pts.append((ahead + r * math.cos(a) + rng.gauss(0, .02), left + r * math.sin(a) + rng.gauss(0, .02), rng.uniform(0.2, 1.6)))
    if wall_y is not None:
        c, s = math.cos(w.dog[2]), math.sin(w.dog[2])
        for wx in np.arange(w.dog[0] - 4, w.dog[0] + 8, 0.01):
            dx, dy = wx - w.dog[0], wall_y - w.dog[1]
            pts.append((dx * c + dy * s, -dx * s + dy * c, rng.uniform(0.2, 1.5)))
    return np.array(pts, dtype=np.float32).reshape(-1, 3)


def walk(segments):
    """Person's (speed, turn rate rad/s) as a function of time from [(until_t, speed, turn_rate), ...]."""
    def f(t):
        for until, v, om in segments:
            if t < until:
                return v, om
        return 0.0, 0.0
    return f


SCENARIOS = {
    # name: (person start ahead/left of the dog, person heading rel. to dog, path, duration, sim-camera z, sim-camera pitch)
    "straight at 0.7 m/s, starting in position": ((1.3, -0.6), 0.0, walk([(20, 0.7, 0)]), 20, CAM.z, 0.0),
    "straight at 0.5 m/s, starting in position": ((1.3, -0.6), 0.0, walk([(20, 0.5, 0)]), 20, CAM.z, 0.0),
    "starts 2.5 m dead ahead, then walks 0.6 m/s": ((2.5, 0.0), 0.0, walk([(3, 0, 0), (20, 0.6, 0)]), 20, CAM.z, 0.0),
    "stop and go (5 s walk, 4 s stand, 5 s walk)": ((1.3, -0.6), 0.0, walk([(5, 0.6, 0), (9, 0, 0), (14, 0.6, 0)]), 14, CAM.z, 0.0),
    "90 deg turn to the left": ((1.3, -0.6), 0.0, walk([(4, 0.6, 0), (5.6, 0.6, math.pi / 2 / 1.6), (16, 0.6, 0)]), 16, CAM.z, 0.0),
    "90 deg turn to the right": ((1.3, -0.6), 0.0, walk([(4, 0.6, 0), (5.6, 0.6, -math.pi / 2 / 1.6), (16, 0.6, 0)]), 16, CAM.z, 0.0),
    "person walks TOWARD the dog (0.4 m/s)": ((2.5, -0.6), math.pi, walk([(10, 0.4, 0)]), 10, CAM.z, 0.0),
    "faster than the dog can go (1.3 m/s)": ((1.3, -0.6), 0.0, walk([(25, 1.3, 0)]), 25, CAM.z, 0.0),
    "person near a wall, camera 10 cm lower than assumed": ((1.3, -0.6), 0.0, walk([(20, 0.5, 0)]), 20, CAM.z - 0.10, 0.0),
    "camera 5 cm lower and 4 deg more downward than assumed": ((1.3, -0.6), 0.0, walk([(20, 0.6, 0)]), 20, CAM.z - 0.05, math.radians(4)),
}


def run(name: str, verbose: bool = False, side: str = "left", seed: int = 1, lidar: bool = False, wall_y: float | None = None,
        **cfg) -> dict:
    (ahead0, left0), rel_heading, path, dur, cam_z, cam_pitch = SCENARIOS[name]
    rng = random.Random(seed)
    w = World(dog=[0.0, 0.0, 0.0], person=[ahead0, left0, rel_heading], cam_z=cam_z, cam_pitch=cam_pitch)
    if side == "right":
        w.person[1] = -left0
    heeler = follow.Heeler(follow.HeelConfig(side=side, use_lidar=lidar, **cfg))
    heeler.reset()
    tx, ty = heeler.target
    dt, t, next_det = 0.02, 0.0, 0.0
    vel = [0.0, 0.0, 0.0]
    pending: list = []                                    # (time it takes effect, cmd)
    cmd, cmd_at = (0.0, 0.0, 0.0), -1.0
    errs, seen, lost_at, min_d = [], 0, None, 9.0
    frames = 0
    while t < dur:
        if t >= next_det:
            next_det += 1 / DET_HZ
            frames += 1
            box = project(w, rng) if rng.random() > 0.05 else None
            seen += box is not None
            cloud = lidar_cloud(w, rng, wall_y) if lidar and frames % LIDAR_EVERY == 0 else None
            res = heeler.step([box] if box else [], (CAM.height, CAM.width, 3), now=t, cloud=cloud)
            if res.lost and lost_at is None:
                lost_at = t
            pending.append((t + LATENCY, (0.0, 0.0, 0.0) if res.lost else res.cmd))
            if verbose and frames % 8 == 0:
                ahead, left = to_dog_frame(w)
                print(f"  t={t:5.1f}  person ({ahead:+.2f},{left:+.2f})  cmd ({res.cmd[0]:+.2f},{res.cmd[1]:+.2f},{res.cmd[2]:+.2f})  {res.status}")
        while pending and pending[0][0] <= t:
            cmd, cmd_at = pending.pop(0)[1], t
        target = cmd if t - cmd_at < 0.7 else (0.0, 0.0, 0.0)   # go2.py's FOLLOW_STALE
        for i in range(3):
            vel[i] += (target[i] - vel[i]) * dt / TAU
        c, s = math.cos(w.dog[2]), math.sin(w.dog[2])
        w.dog[0] += (vel[0] * c - vel[1] * s) * dt
        w.dog[1] += (vel[0] * s + vel[1] * c) * dt
        w.dog[2] += vel[2] * dt
        v, om = path(t)
        w.person[0] += v * math.cos(w.person[2]) * dt
        w.person[1] += v * math.sin(w.person[2]) * dt
        w.person[2] += om * dt
        # (the person's heading is relative to the dog's INITIAL heading, which is the world x axis)
        ahead, left = to_dog_frame(w)
        min_d = min(min_d, math.hypot(ahead, left))
        if t > dur / 2:
            errs.append((abs(ahead - tx), abs(left - ty)))
        t += dt
    errs.sort()
    ex = sorted(e[0] for e in errs)
    ey = sorted(e[1] for e in errs)
    return {"name": name, "seen": seen / max(frames, 1), "ex_med": ex[len(ex) // 2] if ex else float("nan"),
            "ex_p95": ex[int(len(ex) * .95)] if ex else float("nan"),
            "ey_med": ey[len(ey) // 2] if ey else float("nan"), "ey_p95": ey[int(len(ey) * .95)] if ey else float("nan"),
            "min_dist": min_d, "lost_at": lost_at}


def main() -> int:
    verbose = "-v" in sys.argv
    print(f"{'scenario':58s} {'seen':>5s}  {'ahead err m (med/p95)':>22s}  {'side err m (med/p95)':>21s}  {'closest':>7s}  gave up")
    for name in SCENARIOS:
        if verbose:
            print(name)
        r = run(name, verbose)
        print(f"{name:58s} {r['seen']:5.0%}  {r['ex_med']:10.2f} /{r['ex_p95']:6.2f}  {r['ey_med']:10.2f} /{r['ey_p95']:6.2f}  "
              f"{r['min_dist']:6.2f}m  " + (f"at {r['lost_at']:.1f} s" if r["lost_at"] else "no"))
    print("\nSame walks with the lidar sharpening the distance (the camera still decides who and where)")
    print(f"{'scenario':58s} {'seen':>5s}  {'ahead err m (med/p95)':>22s}  {'side err m (med/p95)':>21s}  {'closest':>7s}  gave up")
    for name in SCENARIOS:
        if "camera 5 cm" in name:
            continue
        r = run(name, lidar=True, wall_y=-1.0 if "wall" in name else None)
        print(f"{name:58s} {r['seen']:5.0%}  {r['ex_med']:10.2f} /{r['ex_p95']:6.2f}  {r['ey_med']:10.2f} /{r['ey_p95']:6.2f}  "
              f"{r['min_dist']:6.2f}m  " + (f"at {r['lost_at']:.1f} s" if r["lost_at"] else "no"))
    r = run("straight at 0.7 m/s, starting in position", lidar=True, wall_y=-1.5)
    print(f"{'(distractor) straight walk, wall 0.9 m behind you, lidar on':58s} {r['seen']:5.0%}  {r['ex_med']:10.2f} /{r['ex_p95']:6.2f}  "
          f"{r['ey_med']:10.2f} /{r['ey_p95']:6.2f}  {r['min_dist']:6.2f}m  " + (f"at {r['lost_at']:.1f} s" if r["lost_at"] else "no"))
    r = run("straight at 0.7 m/s, starting in position", side="right")
    print(f"{'(mirror check) same walk, dog on the person\'s right':58s} {r['seen']:5.0%}  {r['ex_med']:10.2f} /{r['ex_p95']:6.2f}  "
          f"{r['ey_med']:10.2f} /{r['ey_p95']:6.2f}  {r['min_dist']:6.2f}m  " + (f"at {r['lost_at']:.1f} s" if r["lost_at"] else "no"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
