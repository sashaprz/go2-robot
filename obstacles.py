"""Obstacle checks from the Go2's lidar point cloud (pure numpy: no dog needed to test it).

The dog publishes its lidar map as points in a fixed WORLD frame plus, separately, its own pose in that frame. These
functions move the points into the dog's frame (x ahead, y left, z up from the floor) and answer questions like
"how far is the nearest thing in the corridor I'm about to walk down?". clearance / sector_ranges / PathWatcher are the building
blocks for a guide mode ("sit when something enters the path") and are NOT wired into go2.py yet. refine_range IS used by heel: the
camera decides who and where the person is, and the lidar may only sharpen the distance along that line of sight. Check what the
real lidar delivers with lidar_probe.py.

  python obstacles.py     runs the self-test on synthetic clouds
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STAND_HEIGHT = 0.32          # m, body height above the floor while standing: the floor is (pose z - this)


def to_dog_frame(points_world: np.ndarray, x: float, y: float, z: float, yaw: float,
                 stand_height: float = STAND_HEIGHT) -> np.ndarray:
    """(N, 3) world points -> (N, 3) in the dog's frame: x ahead, y left, z up from the floor."""
    p = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    dx, dy = p[:, 0] - x, p[:, 1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack((dx * c + dy * s, -dx * s + dy * c, p[:, 2] - (z - stand_height)), axis=1)


@dataclass(frozen=True)
class Corridor:
    """The strip in front of the dog (relative to `heading`, rad, + = left) that must stay clear for it to walk on."""
    length: float = 2.0          # m ahead
    half_width: float = 0.45     # m each side of the centre line: dog (~0.3 m) plus the person it leads
    z_min: float = 0.12          # m above the floor: ignore the floor itself and its noise...
    z_max: float = 1.0           # ...and anything higher than the dog + person need (overhangs are a separate problem)
    min_points: int = 4          # a lone stray point is noise, not an obstacle
    heading: float = 0.0


def clearance(pts: np.ndarray, c: Corridor = Corridor()) -> float | None:
    """Distance (m) to the nearest obstacle inside the corridor, or None when the corridor is clear. pts: dog frame."""
    if len(pts) == 0:
        return None
    ch, sh = math.cos(-c.heading), math.sin(-c.heading)
    along = pts[:, 0] * ch - pts[:, 1] * sh                  # coordinates in the corridor's own frame
    across = pts[:, 0] * sh + pts[:, 1] * ch
    hit = (along > 0.0) & (along < c.length) & (np.abs(across) < c.half_width) & (pts[:, 2] > c.z_min) & (pts[:, 2] < c.z_max)
    if int(hit.sum()) < c.min_points:
        return None
    return float(np.percentile(along[hit], 5))                # the near edge, robust to a stray point


def sector_ranges(pts: np.ndarray, max_range: float = 3.0, z_min: float = 0.12, z_max: float = 1.0) -> dict[str, float]:
    """Nearest obstacle in each 90 degree sector around the dog (front / left / back / right), max_range if none."""
    out = {"front": max_range, "left": max_range, "back": max_range, "right": max_range}
    if len(pts) == 0:
        return out
    m = (pts[:, 2] > z_min) & (pts[:, 2] < z_max)
    q = pts[m]
    d = np.hypot(q[:, 0], q[:, 1])
    ang = np.degrees(np.arctan2(q[:, 1], q[:, 0]))              # 0 = ahead, + = left
    for name, lo, hi in (("front", -45, 45), ("left", 45, 135), ("right", -135, -45)):
        sel = (ang >= lo) & (ang < hi) & (d < max_range)
        if sel.sum() >= 4:
            out[name] = float(np.percentile(d[sel], 5))
    sel = ((ang >= 135) | (ang < -135)) & (d < max_range)
    if sel.sum() >= 4:
        out["back"] = float(np.percentile(d[sel], 5))
    return out


def refine_range(pts: np.ndarray, lens: tuple[float, float], bearing: float, s_cam: float, window: float = 0.6,
                 half_width: float = 0.3, z_min: float = 0.2, z_max: float = 1.3, min_points: int = 10,
                 body: tuple[float, float] = (0.35, 0.25)) -> float | None:
    """Sharpen the camera's distance estimate with the lidar. The CAMERA decides who and in which direction: the person
    is on the ray from `lens` (x, y in the dog frame) at `bearing` (rad, + = left), about `s_cam` m away. Here we look
    along that ray, within `window` m of that range, for the NEAREST compact blob at legs-to-chest height and return
    its range in m from the lens, or None to keep the camera's number.

    Refuses (None) when the blob keeps going sideways (a wall, a sofa: a person doesn't), when there are too few
    points, or when nothing sits inside the window. It can only move the estimate along the camera's line of sight,
    and by at most `window`, so it can never pull the dog toward something the camera doesn't see."""
    if len(pts) == 0:
        return None
    u = np.array([math.cos(bearing), math.sin(bearing)], dtype=np.float32)
    n = np.array([-u[1], u[0]], dtype=np.float32)
    rel = pts[:, :2] - np.asarray(lens, dtype=np.float32)
    s, d = rel @ u, rel @ n
    h = pts[:, 2]
    band = (h > z_min) & (h < z_max) & ~((np.abs(pts[:, 0]) < body[0]) & (np.abs(pts[:, 1]) < body[1]))
    cand = band & (np.abs(d) < half_width) & (np.abs(s - s_cam) < window)
    if int(cand.sum()) < min_points:
        return None
    ss = np.sort(s[cand])
    start = None
    for i in range(len(ss) - min_points + 1):                    # the nearest patch of min_points within 0.45 m of range
        if ss[i + min_points - 1] - ss[i] < 0.45:
            start = i
            break
    if start is None:
        return None
    blob = ss[(ss >= ss[start]) & (ss < ss[start] + 0.45)]
    s_lid = float(np.median(blob))
    beside = band & (np.abs(d) >= half_width) & (np.abs(d) < 0.9) & (np.abs(s - s_lid) < 0.3)
    if int(beside.sum()) > 0.5 * len(blob):                      # it continues sideways: not a person
        return None
    return s_lid


class PathWatcher:
    """Turns a stream of clearance readings into 'the path is blocked', without flicker.

    Blocked as soon as something is within `stop_at`; only clear again after `clear_for` seconds of the corridor being
    empty. A guide should be quick to stop and slow to set off again."""

    def __init__(self, stop_at: float = 1.2, clear_for: float = 1.5):
        self.stop_at, self.clear_for = stop_at, clear_for
        self.blocked = False
        self._clear_since: float | None = None

    def update(self, dist: float | None, now: float) -> bool:
        if dist is not None and dist <= self.stop_at:
            self.blocked, self._clear_since = True, None
        elif self.blocked:
            self._clear_since = now if self._clear_since is None else self._clear_since
            if now - self._clear_since >= self.clear_for:
                self.blocked, self._clear_since = False, None
        return self.blocked


# ---- self-test on synthetic clouds ------------------------------------------------------------------------
def _box(cx, cy, w, d, h, n=400, rng=np.random.default_rng(0)):
    """Points on the surface-ish of an upright box standing on the floor (world coordinates)."""
    return np.stack((cx + (rng.random(n) - .5) * w, cy + (rng.random(n) - .5) * d, rng.random(n) * h), axis=1)


def _selftest() -> int:
    rng = np.random.default_rng(1)
    floor = np.stack((rng.uniform(-4, 4, 3000), rng.uniform(-4, 4, 3000), rng.normal(0, 0.01, 3000)), axis=1)
    checks = {}

    def look(cloud, x=0.0, y=0.0, yaw=0.0, corridor=Corridor()):
        return clearance(to_dog_frame(np.vstack((floor, cloud)), x, y, STAND_HEIGHT, yaw), corridor)

    checks["floor alone is not an obstacle"] = look(np.zeros((0, 3))) is None
    d = look(_box(1.5, 0.0, 0.3, 0.3, 0.6))
    checks[f"box 1.5 m ahead is seen at about 1.35 m (got {d})"] = d is not None and abs(d - 1.35) < 0.15
    checks["box beside the path (0.9 m to the left) is ignored"] = look(_box(1.5, 0.9, 0.3, 0.3, 0.6)) is None
    checks["box behind the dog is ignored"] = look(_box(-1.5, 0.0, 0.3, 0.3, 0.6)) is None
    checks["box further than the corridor is ignored"] = look(_box(3.0, 0.0, 0.3, 0.3, 0.6)) is None
    checks["a 5 cm ridge on the floor is ignored"] = look(_box(1.0, 0.0, 0.5, 0.5, 0.05)) is None
    checks["two stray points are noise"] = look(np.array([[1.0, 0.0, 0.5], [1.1, 0.0, 0.4]])) is None
    d = look(_box(0.0, 1.5, 0.3, 0.3, 0.6), x=0.0, y=0.0, yaw=math.pi / 2)         # dog turned to face +y: box is ahead
    checks[f"the dog's heading is honoured (turned 90 deg, box now ahead; got {d})"] = d is not None and abs(d - 1.35) < 0.15
    checks["...and the same box is NOT ahead when the dog faces +x"] = look(_box(0.0, 1.5, 0.3, 0.3, 0.6), yaw=0.0) is None
    d = look(_box(3.0, 2.0, 0.3, 0.3, 0.6), x=2.0, y=2.0, yaw=0.0)                  # dog moved to (2, 2): box 1 m ahead
    checks[f"the dog's position is honoured (got {d})"] = d is not None and abs(d - 0.85) < 0.15
    d = look(_box(1.5, 1.5, 0.3, 0.3, 0.6), corridor=Corridor(heading=math.radians(45)))
    checks[f"a corridor aimed 45 deg left sees a box up the diagonal (got {d})"] = d is not None and abs(d - 2.0) < 0.3
    r = sector_ranges(to_dog_frame(np.vstack((floor, _box(0.0, 0.8, 0.3, 0.3, 0.7))), 0, 0, STAND_HEIGHT, 0.0))
    checks[f"a person-sized box 0.8 m to the LEFT shows up in the left sector only ({r})"] = (
        r["left"] < 1.0 and r["front"] == 3.0 and r["right"] == 3.0 and r["back"] == 3.0)
    lens = (0.3, 0.0)
    person = _box(2.0, -0.5, 0.4, 0.4, 1.6, n=120)                   # 1.7 m ahead of the lens, 0.5 m to the right
    br = math.atan2(-0.5, 2.0 - lens[0])
    sc = math.hypot(2.0 - lens[0], 0.5)
    d = lambda cloud: refine_range(to_dog_frame(np.vstack((floor, cloud)), 0, 0, STAND_HEIGHT, 0.0), lens, br, sc)  # noqa: E731
    r = d(person)
    checks[f"a person on the camera's line of sight is found at the right range (got {r}, want {sc:.2f})"] = r is not None and abs(r - sc) < 0.12
    r = refine_range(to_dog_frame(np.vstack((floor, person)), 0, 0, STAND_HEIGHT, 0.0), lens, br, sc + 0.4)
    checks[f"a camera range that is 0.4 m too long is pulled back to the person (got {r})"] = r is not None and abs(r - sc) < 0.12
    wall = np.stack((np.arange(-1, 6, 0.02), np.full(350, -1.2), np.random.default_rng(2).uniform(0.0, 1.8, 350)), axis=1)
    checks["a wall alone (camera range slightly off) is refused: it continues sideways"] = d(wall) is None
    r = d(np.vstack((person, wall)))
    checks[f"a person 0.7 m in front of a wall gets the PERSON's range, not the wall's (got {r})"] = r is not None and abs(r - sc) < 0.12
    checks["nothing near the camera's range: keeps the camera's number"] = refine_range(
        to_dog_frame(np.vstack((floor, _box(5.0, 3.0, 0.4, 0.4, 1.6))), 0, 0, STAND_HEIGHT, 0.0), lens, br, sc) is None
    checks["a 15 cm kerb is not a person"] = d(_box(2.0, -0.5, 0.5, 0.5, 0.15)) is None
    checks["the dog's own body is not a person"] = refine_range(
        to_dog_frame(np.vstack((floor, _box(0.0, 0.0, 0.3, 0.2, 0.5, n=120))), 0, 0, STAND_HEIGHT, 0.0), lens, 0.0, 0.3, window=0.6) is None
    w = PathWatcher(stop_at=1.2, clear_for=1.5)
    seq = [(0.0, 3.0), (0.1, 1.1), (0.2, None), (1.0, None), (1.7, None), (1.8, 1.0), (2.0, None), (3.6, None)]
    got = [w.update(dist, t) for t, dist in seq]
    checks[f"the watcher stops at once, re-arms after 1.5 s clear, stops again at once ({got})"] = got == [False, True, True, True, False, True, True, False]
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
