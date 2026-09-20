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
    "person walks straight AT the dog, dead ahead (0.4 m/s)": ((2.5, 0.0), math.pi, walk([(10, 0.4, 0)]), 10, CAM.z, 0.0),
    "faster than the dog can go (1.3 m/s)": ((1.3, -0.6), 0.0, walk([(25, 1.3, 0)]), 25, CAM.z, 0.0),
    "sharp right turn (90 deg in 1 s)": ((1.3, -0.35), 0.0, walk([(4, 0.6, 0), (5.0, 0.6, -math.pi / 2 / 1.0), (16, 0.6, 0)]), 16, CAM.z, 0.0),
    "S-bend: right then left": ((1.3, -0.35), 0.0, walk([(4, 0.6, 0), (5.6, 0.6, -math.pi / 2 / 1.6), (8, 0.6, 0), (9.6, 0.6, math.pi / 2 / 1.6), (18, 0.6, 0)]), 18, CAM.z, 0.0),
    "brisk walk at 1.0 m/s": ((1.3, -0.35), 0.0, walk([(20, 1.0, 0)]), 20, CAM.z, 0.0),
    "walk 8 s at 0.8 m/s, then stand still 8 s": ((1.3, -0.35), 0.0, walk([(8, 0.8, 0)]), 16, CAM.z, 0.0),
    "person near a wall, camera 10 cm lower than assumed": ((1.3, -0.6), 0.0, walk([(20, 0.5, 0)]), 20, CAM.z - 0.10, 0.0),
    "camera 5 cm lower and 4 deg more downward than assumed": ((1.3, -0.6), 0.0, walk([(20, 0.6, 0)]), 20, CAM.z - 0.05, math.radians(4)),
}


def run(name: str, verbose: bool = False, side: str = "left", seed: int = 1, lidar: bool = False, wall_y: float | None = None,
        latency: float | None = None, det_hz: float | None = None, controller: str = "heel", lidar_bias: float = 0.0, lidar_until: float | None = None, cam_override: tuple | None = None, **cfg) -> dict:
    (ahead0, left0), rel_heading, path, dur, cam_z, cam_pitch = SCENARIOS[name]
    rng = random.Random(seed)
    if cam_override:
        cam_z, cam_pitch = cam_override
    w = World(dog=[0.0, 0.0, 0.0], person=[ahead0, left0, rel_heading], cam_z=cam_z, cam_pitch=cam_pitch)
    if side == "right":
        w.person[1] = -left0
    if controller == "follow":                              # follow.Follower with a sideways offset: the simple, image-only way
        heeler = follow.Follower(follow.FollowConfig(**cfg))
        heeler.reset()
        tx, ty = 0.0, 0.0
    else:
        heeler = follow.Heeler(follow.HeelConfig(side=side, use_lidar=lidar, **cfg))
        heeler.reset()
        tx, ty = heeler.target
    latency = LATENCY if latency is None else latency
    det_hz = DET_HZ if det_hz is None else det_hz
    dt, t, next_det = 0.02, 0.0, 0.0
    vel = [0.0, 0.0, 0.0]
    pending: list = []                                    # (time it takes effect, cmd)
    cmd, cmd_at = (0.0, 0.0, 0.0), -1.0
    errs, seen, lost_at, min_d = [], 0, None, 9.0
    yaws, dists, lats, vxs = [], [], [], []
    frames = 0
    while t < dur:
        if t >= next_det:
            next_det += 1 / det_hz
            frames += 1
            box = project(w, rng) if rng.random() > 0.05 else None
            seen += box is not None
            cloud = lidar_cloud(w, rng, wall_y) if lidar and frames % LIDAR_EVERY == 0 and (lidar_until is None or t < lidar_until) else None
            if cloud is not None and lidar_bias:                # the lidar sees the FRONT of the body: nearer than the feet
                cloud[:, 0] -= lidar_bias
            res = heeler.step([box] if box else [], (CAM.height, CAM.width, 3), now=t, cloud=cloud)
            if res.lost and lost_at is None:
                lost_at = t
            yaws.append(res.cmd[2])
            vxs.append(res.cmd[0])
            pending.append((t + latency, (0.0, 0.0, 0.0) if res.lost else res.cmd))
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
            dists.append(math.hypot(ahead, left))
            lats.append(abs(left))
        t += dt
    errs.sort()
    ex = sorted(e[0] for e in errs)
    ey = sorted(e[1] for e in errs)
    return {"name": name, "seen": seen / max(frames, 1), "ex_med": ex[len(ex) // 2] if ex else float("nan"),
            "ex_p95": ex[int(len(ex) * .95)] if ex else float("nan"),
            "ey_med": ey[len(ey) // 2] if ey else float("nan"), "ey_p95": ey[int(len(ey) * .95)] if ey else float("nan"),
            "min_dist": min_d, "lost_at": lost_at, "yaw_rms": float(np.sqrt(np.mean(np.square(yaws)))) if yaws else 0.0,
            "vx_std": float(np.std(vxs[len(vxs) // 2:])) if vxs else 0.0, "dist_med": float(np.median(dists)) if dists else float("nan"), "lat_med": float(np.median(lats)) if lats else float("nan")}


def unit_checks() -> int:
    """The lidar 'someone is beside me' wait, and the loss reasons."""
    rng = random.Random(5)
    frame = (CAM.height, CAM.width, 3)
    checks = {}

    def run_case(cloud_fn, label):
        h = follow.Heeler(follow.HeelConfig(use_lidar=True))
        h.reset()
        w = World(dog=[0.0, 0.0, 0.0], person=[1.3, -0.35, 0.0])
        for i in range(8):                                     # the camera sees them for a while ...
            h.step([project(w, rng)], frame, now=i * 0.07, cloud=cloud_fn(w))
        out, t = [], 8 * 0.07
        w.person[:2] = [0.2, -0.45]                            # ... then they end up level with the dog: out of the camera's view
        for k in (0.4, 1.0, 3.0, 5.5, 7.0):
            r = h.step([], frame, now=t + k, cloud=cloud_fn(w))
            out.append((k, r))
        return out

    beside = run_case(lambda w: lidar_cloud(w, rng), "person beside")
    checks["camera loses them but the lidar sees someone beside: the dog waits (no movement) for up to 6 s"] = all(
        r.status.startswith("waiting") and r.cmd == (0.0, 0.0, 0.0) and not r.lost for k, r in beside if 0.4 <= k <= 5.5)
    checks["...then gives up if they never come back into view"] = beside[-1][1].lost
    gone = run_case(lambda w: lidar_cloud(w, rng, wall_y=-0.7) if False else np.zeros((0, 3), np.float32), "nobody")
    checks["camera loses them and the lidar sees nobody: gives up after 1.5 s, saying why"] = (
        gone[3][1].lost and "no person detected" in gone[3][1].status)

    def wall_only(w):
        far = World(dog=w.dog, person=[9.0, 9.0, 0.0])         # nobody near the dog, but a long wall where they were
        return lidar_cloud(far, rng, wall_y=-0.7)
    wall = run_case(wall_only, "wall")
    checks["a wall where they were does NOT make the dog wait: it gives up as usual"] = wall[3][1].lost
    # camera calibration from people standing at two known distances
    def feet_row(z, tilt_deg, d, noise):
        w = World(dog=[0, 0, 0], person=[CAM.x_off + d, 0.0, 0.0], cam_z=z, cam_pitch=math.radians(tilt_deg))
        return project(w, rng, noise=noise)[3]

    def worst_distance_error(noise, trials=20):
        """Height and tilt trade off against each other, so judge the fit by what the dog uses: the distance it then predicts."""
        worst, refused = 0.0, 0
        for z_true, t_true in ((0.35, 0.0), (0.31, 5.0), (0.38, 8.0), (0.28, 2.0), (0.34, 12.0)):
            for _ in range(trials):
                r = follow.solve_camera([(feet_row(z_true, t_true, d, noise), d) for d in (1.2, 2.4)], frame_h=CAM.height)
                if r is None:
                    refused += 1
                    continue
                fitted = follow.Camera(z=r[0], pitch=r[1])
                for d in (0.9, 1.2, 1.6, 2.0):                                 # judged over the range heel actually works in (beyond ~2 m 1 px is 10+ cm)
                    row = feet_row(z_true, t_true, d, 0.0)
                    box = (600, 100, 700, row, 0.9)
                    worst = max(worst, abs(follow.locate(box, (720, 1280, 3), fitted).x - CAM.x_off - d))
        return worst, refused

    single, r1 = worst_distance_error(4.0)                    # one noisy frame per spot
    averaged, r2 = worst_distance_error(1.0)                  # the app averages ~20 frames per spot
    print(f"  calibration from 2 spots: worst distance error {single * 100:.0f} cm (one frame, +-4 px jitter), {averaged * 100:.0f} cm (averaged), refused {r1 + r2}")
    checks["calibrated distances are within 16 cm everywhere from 0.9 to 2 m even from single noisy frames"] = single < 0.16 and r1 == 0
    checks["...and within 6 cm when the box jitter is averaged, as the app does"] = averaged < 0.06 and r2 == 0
    exact = follow.solve_camera([(feet_row(0.31, 5.0, d, 0.0), d) for d in (1.2, 2.4)], frame_h=CAM.height)
    checks[f"noise-free it is exact (got {exact and (round(exact[0], 3), round(math.degrees(exact[1]), 2))}, want 0.31 m and 5 deg)"] = (
        exact is not None and abs(exact[0] - 0.31) < 0.005 and abs(math.degrees(exact[1]) - 5.0) < 0.3)
    one = follow.solve_camera([(feet_row(0.31, 0.0, 1.5, 0.0), 1.5)], frame_h=CAM.height)
    checks["one spot (assumes a level camera) also works when the camera really is level"] = one is not None and abs(one[0] - 0.31) < 0.005
    checks["a mistaken measurement (same feet row at two different distances) is refused, not applied"] = (
        follow.solve_camera([(500, 1.2), (500, 2.4)], frame_h=CAM.height) is None)
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


def plain_follow_checks() -> int:
    """Normal 'follow me' with the closer stop point and higher speed (follow.follow_config), against the old settings."""
    import dataclasses

    new = dict(dataclasses.asdict(follow.follow_config()), controller="follow", x_offset=0.0)
    old = dict(controller="follow", x_offset=0.0, lost_after=1.2, target_height=0.60, max_forward=0.35, k_forward=1.0)
    print()
    print("normal follow, old settings (stop at 60% of the picture, 0.35 m/s) against the new (78%, 0.8 m/s), median distance to the person:")
    print(f"  {'scenario':44s} {'old':>14s} {'new':>14s}   lost (new, fast / 10 fps link)")
    lost_new = 0
    stand_new = 9.0
    for name in ("walk 8 s at 0.8 m/s, then stand still 8 s", "straight at 0.5 m/s, starting in position", "straight at 0.7 m/s, starting in position",
                 "stop and go (5 s walk, 4 s stand, 5 s walk)", "90 deg turn to the left", "90 deg turn to the right", "S-bend: right then left"):
        ro = [run(name, seed=sd, **old) for sd in (1, 2, 3)]
        rn = [run(name, seed=sd, **new) for sd in (1, 2, 3)]
        rm = [run(name, seed=sd, latency=0.22, det_hz=10.0, **new) for sd in (1, 2, 3)]
        lf, lm = sum(r["lost_at"] is not None for r in rn), sum(r["lost_at"] is not None for r in rm)
        lost_new += lf + lm
        if name.startswith("walk 8 s"):
            stand_new = sum(r["dist_med"] for r in rn) / 3
        print(f"  {name:44s} {sum(r['dist_med'] for r in ro) / 3:11.2f} m {sum(r['dist_med'] for r in rn) / 3:11.2f} m   {lf}/3, {lm}/3")
    at_dog = min(run("person walks straight AT the dog, dead ahead (0.4 m/s)", seed=sd, **new)["min_dist"] for sd in (1, 2, 3))
    at_dog_old = min(run("person walks straight AT the dog, dead ahead (0.4 m/s)", seed=sd, **old)["min_dist"] for sd in (1, 2, 3))
    print(f"  a person walking straight at the dog gets no closer than {at_dog:.2f} m (old settings: {at_dog_old:.2f} m)")
    checks = {
        "the new follow never lost the person (7 scenarios x 3 runs, fast and 10 fps links)": lost_new == 0,
        "it stops closer: about 1.3 m standing (the old settings were still 5+ m away after the same walk)": stand_new < 1.6,
        "a person walking straight at the dog is kept at least 0.8 m away": at_dog >= 0.8,
    }
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


def follow_style_checks() -> int:
    """The DEFAULT heel: the follow controller held off-centre, the lidar teaching the camera the distance (follow.heel_follow_config)."""
    import dataclasses

    cfg = dict(dataclasses.asdict(follow.heel_follow_config(max_forward=1.2, range_target=1.1)), controller="follow", lidar=True)
    links = {"fast (0.15 s, 15 fps)": {}, "medium (0.22 s, 10 fps)": dict(latency=0.22, det_hz=10.0), "slow (0.3 s, 8 fps)": dict(latency=0.3, det_hz=8.0)}
    print()
    print("the default heel (camera steers, the lidar teaches the camera the distance, target 1.1 m), 3 random runs per cell:")
    print(f"  {'scenario':44s} " + " ".join(f"{k:>30s}" for k in links))
    lost_fast = lost_medium = 0
    for name in ("walk 8 s at 0.8 m/s, then stand still 8 s", "straight at 0.7 m/s, starting in position", "brisk walk at 1.0 m/s",
                 "stop and go (5 s walk, 4 s stand, 5 s walk)", "90 deg turn to the left", "90 deg turn to the right",
                 "sharp right turn (90 deg in 1 s)", "S-bend: right then left"):
        cells = []
        for k, kw in links.items():
            rs = [run(name, seed=sd, **cfg, **kw) for sd in (1, 2, 3)]
            lost = sum(r["lost_at"] is not None for r in rs)
            if k.startswith("fast"):
                lost_fast += lost
            elif k.startswith("medium"):
                lost_medium += lost
            cells.append(f"lost {lost}/3, {sum(r['dist_med'] for r in rs) / 3:.2f} m from you")
        print(f"  {name:44s} " + " ".join(f"{c:>30s}" for c in cells))
    W = math.radians
    mis = [run("walk 8 s at 0.8 m/s, then stand still 8 s", seed=sd, **cfg, cam_override=(0.25, W(6))) for sd in (1, 2, 3)]
    mis_d = sum(r["dist_med"] for r in mis) / 3
    print(f"  camera badly mis-set (10 cm low, 6 deg tilt), lidar on, fast link: standing {mis_d:.2f} m from you, lost {sum(r['lost_at'] is not None for r in mis)}/3")
    walls = [run("straight at 0.7 m/s, starting in position", seed=sd, **cfg, wall_y=-1.0, lidar_bias=0.10) for sd in (1, 2, 3)]
    print(f"  wall 0.4 m behind you AND the lidar reading the front of your body: walking {sum(r['dist_med'] for r in walls) / 3:.2f} m, lost {sum(r['lost_at'] is not None for r in walls)}/3")
    toward = min(run("person walks straight AT the dog, dead ahead (0.4 m/s)", seed=sd, **cfg)["min_dist"] for sd in (1, 2, 3))
    print(f"  a person walking straight at the dog, dead ahead, gets no closer to it than {toward:.2f} m")
    checks = {
        "fast link (15 fps): never lost the person in any start, stop-and-go or turn": lost_fast == 0,
        f"medium link (10 fps): regression guard, at most 3 of 24 runs lost (now {lost_medium}); the slow link (8 fps) is worse and only reported": lost_medium <= 3,
        "holds you close: about 1.1 m standing even when the camera is badly mis-set (the lidar taught it)": 0.9 <= mis_d <= 1.3 and all(r["lost_at"] is None for r in mis),
        "a wall behind you and a lidar that reads the front of your body change nothing": all(r["lost_at"] is None for r in walls) and sum(r["dist_med"] for r in walls) / 3 < 1.8,
        "a person walking straight at the dog, dead ahead, is kept at least 0.6 m away (it backs off; without it they reached 0.11 m)": toward >= 0.6,
    }
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


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
    print("\nturns on a slower link (0.3 s lag, 8 fps detector), 3 random runs each; 'lost' = the dog gave up on the person:")
    slow_lost = 0
    for name in ("90 deg turn to the left", "90 deg turn to the right", "sharp right turn (90 deg in 1 s)", "S-bend: right then left"):
        lost = sum(run(name, seed=sd, latency=0.3, det_hz=8.0)["lost_at"] is not None for sd in (1, 2, 3))
        slow_lost += lost
        print(f"  {name:44s} lost {lost}/3")
    print("\nlidar wait + loss reasons:")
    rc = follow_style_checks() | plain_follow_checks() | unit_checks()
    print(f"  {'PASS' if slow_lost == 0 else 'FAIL'}  the dog kept hold of the person through every turn on the slower link ({slow_lost} of 12 runs lost them)")
    return rc or (1 if slow_lost else 0)


if __name__ == "__main__":
    sys.exit(main())
