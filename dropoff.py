"""Drop-off detection (stairs down, a kerb, a ledge) from the Go2's lidar. numpy + scipy.ndimage: no dog needed to test it.

The lidar sits ~0.35 m above the floor, so it cannot see the ground just beyond a ledge: the lip hides it. What it does see is
  (1) a VOID: floor that was returned right up to some line and then is not, in plain line of sight (nothing standing in front
      of it that could be casting a shadow). A step down of depth h seen from distance D hides a strip about D*h/0.35 long.
  (2) LOW floor: returns well below the floor (the lower steps, the ground beneath), when the geometry lets it see them.
Either one is a cliff. It says nothing about stairs going UP (their risers are ordinary obstacles) or about ramps.

What the real recording (lidar_recording.npz) fixed:
  - floor cells are returned 100% out to 1.5 m ahead, ~80% at 2-3 m, ~30% at 3-4 m, and poorly behind the dog or beyond a wall
    -> only look in the FRONT +-85 degrees, between 0.5 and 2.5 m
  - flat floor scatters within about -0.08 .. +0.04 m and NOTHING was below -0.12 m -> DROP_Z = -0.12
  - so a void only counts when the floor just before it (0.2-0.6 m nearer, along the line of sight) WAS seen, and it is a compact patch
    (>= 20 cells, >= 0.3 m across) with at least 0.6 m of well-supported lip, and only if the floor 0.5-1.0 m ahead is >= 70% returned at all
    (the real dog: 100%): sparse coverage, or a lidar that isn't seeing that way, does not make a cliff.
A ledge only shows if its shadow is at least 3 cells (0.3 m) deep: from 2 m that is a drop of about 5 cm or more, less than that is not seen.
A dark or mirrored floor looks like a void (it stops: the safe way to be wrong). It only judges what is between 0.5 and 2.5 m, so callers
must remember what they saw (pathplan.RoomMap does), and a ledge is first noticed about 2.1 m away.

  python dropoff.py      runs the self-test on synthetic scenes (a step, a staircase, a box's shadow, sparse floor ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from obstacles import HEAD_TOP

DROP_Z = -0.12               # m, dog frame (floor = 0): returns lower than this are the ground below a ledge
LIDAR_H = 0.35               # m above the floor: only used by the self-test's scene generator


@dataclass
class Result:
    cliff: np.ndarray            # bool, same shape as the grids: cells that are a drop-off
    window: np.ndarray           # bool: the cells this call judged; the caller keeps its memory of the rest
    nearest: float | None        # m from the dog to the nearest cliff cell, None if there is none


def grids_from_points(pts: np.ndarray, res: float = 0.1, extent=(-1.0, -3.5, 3.5, 3.5), drop_z: float = DROP_Z,
                      z_lo: float = 0.15, z_hi: float = HEAD_TOP, min_hits: int = 3):
    """One message of dog-frame points -> (seen, low, obstacle, origin) grids for detect(). For callers with no map of their own
    (corridor.py). extent = (x0, y0, x1, y1) in the dog frame; origin is the corner of cell [0, 0]."""
    x0, y0, x1, y1 = extent
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    ix = np.floor((pts[:, 0] - x0) / res).astype(np.int64)
    iy = np.floor((pts[:, 1] - y0) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat, z = iy[ok] * nx + ix[ok], pts[ok, 2]

    def count(mask):
        return np.bincount(flat[mask], minlength=nx * ny).reshape(ny, nx)
    seen = count(z <= z_hi) > 0
    low = count((z < drop_z) & (z > -1.5))
    obst = count((z >= z_lo) & (z <= z_hi)) >= min_hits
    return seen, low, obst, (x0, y0)


def detect(seen: np.ndarray, low: np.ndarray, obst: np.ndarray, origin, res: float, pose, near: float = 0.5, far: float = 2.5,
           half_fov: float = math.radians(85), min_cells: int = 20, min_span: int = 3, support: float = 0.7, min_low: int = 2,
           min_low_cells: int = 3, min_edge: int = 6, near_cover: float = 0.7) -> Result:
    """Find drop-offs on world- or dog-aligned grids [iy, ix] whose cell [0, 0] has its corner at `origin`. pose = (x, y, yaw) of the dog in
    the same frame. seen: any return at all in the cell; low: number of returns below DROP_Z; obst: an obstacle stands there."""
    ny, nx = seen.shape
    xc = origin[0] + (np.arange(nx) + 0.5) * res
    yc = origin[1] + (np.arange(ny) + 0.5) * res
    dx, dy = xc[None, :] - pose[0], yc[:, None] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    u, v = dx * c + dy * s, -dx * s + dy * c                      # ahead, left
    r, bear = np.hypot(u, v), np.arctan2(v, u)
    window = (r >= near) & (r <= far) & (np.abs(bear) <= half_fov)
    cliff = np.zeros_like(window)

    # shadows: a void with an obstacle nearer along (nearly) the same bearing is hidden behind it, not a drop
    nb = 360
    bins = np.clip(np.floor(np.degrees(bear) + 180.0).astype(np.int64), 0, nb - 1)
    obs_r = np.full(nb, np.inf)
    if obst.any():
        np.minimum.at(obs_r, bins[obst], r[obst])
        obs_r = np.min([np.roll(obs_r, k) for k in range(-3, 4)], axis=0)       # +-3 degrees
    # a void is judged by LOCAL DENSITY, not cell by cell: earlier views fill parts of the shadow and ~8% of floor cells are missing anyway, so the
    # raw unseen cells are ragged. 55% of the 5x5 around a cell unseen (and not in an obstacle's shadow) makes it void; a thin strip (a chair leg's
    # shadow, 2 cells wide: 40%) does not.
    unseen = ~seen & (r <= obs_r[bins])
    void = window & (ndimage.uniform_filter(unseen.astype(np.float32), size=5, mode="constant") >= 0.55)
    ring = window & (r < 1.0) & (np.abs(bear) <= math.radians(60))
    if ring.sum() < 10 or seen[ring].mean() < near_cover:      # the real dog sees 100% of the floor there; if this one doesn't, don't trust voids
        void[:] = False

    if void.any():
        lab, n = ndimage.label(void, structure=np.ones((3, 3)))
        iy, ix = np.nonzero(void)
        rr = r[iy, ix]
        def before(dist):                                          # the cell `dist` m nearer the dog along the line of sight: (ok, iy, ix)
            ax = xc[ix] - dist * (xc[ix] - pose[0]) / rr
            ay = yc[iy] - dist * (yc[iy] - pose[1]) / rr
            bx = np.floor((ax - origin[0]) / res).astype(np.int64)
            by = np.floor((ay - origin[1]) / res).astype(np.int64)
            ok = (bx >= 0) & (bx < nx) & (by >= 0) & (by < ny)
            return ok, np.clip(by, 0, ny - 1), np.clip(bx, 0, nx - 1)
        ok1, y1, x1 = before(0.3)
        edge = ~void[y1, x1]                                       # a cell of the void's near edge
        # ...and it only counts if the floor just before it WAS returned (at least 6 of the 7 places 0.2-0.8 m nearer, ignoring any that fall
        # under the dog itself): that is what tells "the lidar sees the floor here, and then it stops" from "the lidar is not seeing much
        # floor here". Coverage of 59% (what a sparse scan gives at 2 m) has 6 of 7 seen ~13% of the time; the real floor is 90-100% seen
        # out to 2.5 m.
        seen_before = np.zeros(len(iy), int)
        usable = np.zeros(len(iy), int)
        for dist in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            okd, yd, xd = before(dist)
            away = okd & (np.hypot(xc[xd] - pose[0], yc[yd] - pose[1]) >= 0.45)
            usable += away
            seen_before += away & seen[yd, xd]
        good = edge & (usable >= 5) & (seen_before >= usable - 1)
        lbl = lab[iy, ix]
        n_edge = np.bincount(lbl[edge], minlength=n + 1)
        n_sup = np.bincount(lbl[good], minlength=n + 1)
        size = np.bincount(lab.ravel(), minlength=n + 1)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if size[k] >= min_cells and h >= min_span and w >= min_span and n_edge[k] >= min_edge and n_sup[k] >= max(min_edge, support * n_edge[k]):
                cliff |= lab == k

    lowc = window & (low >= min_low)                               # the ground below the ledge, seen
    if lowc.any():
        lab, n = ndimage.label(lowc, structure=np.ones((3, 3)))
        size = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.nonzero(size >= min_low_cells)[0]
        cliff |= np.isin(lab, keep[keep > 0])

    nearest = float(r[cliff].min()) if cliff.any() else None
    return Result(cliff, window, nearest)


def cells_xy(res: Result, origin, cell: float) -> np.ndarray:
    """Centres (N, 2) of the cliff cells, in the grid's frame."""
    iy, ix = np.nonzero(res.cliff)
    return np.stack((origin[0] + (ix + 0.5) * cell, origin[1] + (iy + 0.5) * cell), axis=1)


# ---- self-test: synthetic scenes ---------------------------------------------------------------------------------
def _scene(edge=1.8, depth=0.17, steps=1, run=0.3, angle=0.0, density=350.0, noise=0.03, box=None, wall=None, seed=0, x_max=5.0):
    """Dog-frame points (x ahead, y left) of a flat floor with a ledge whose line is at `edge` m along the direction `angle` (rad), stepping
    down `depth` m per step, `steps` steps `run` m apart. Floor beyond a ledge is only returned where the lip does not hide it from a lidar
    at LIDAR_H. box = (x0, x1, half_width, height): an upright box, whose floor shadow is hidden. wall = x: a wall across the view."""
    rng = np.random.default_rng(seed)
    n = int(density * (x_max + 1.0) * 7.0)
    x, y = rng.uniform(-1.0, x_max, n), rng.uniform(-3.5, 3.5, n)
    nrm = np.array([math.cos(angle), math.sin(angle)])
    d = x * nrm[0] + y * nrm[1]                                    # distance along the ledge's normal
    edges = [edge + k * run for k in range(steps)]
    level = np.zeros(n, int)
    for k, e in enumerate(edges, start=1):
        level[d >= e] = k
    z = -depth * level + rng.normal(0.0, noise, n)
    seen = np.ones(n, bool)
    for k in range(1, steps + 1):                                  # a point on step k must be visible over the lips of the k steps in front of it
        m = level == k
        for j in range(1, k + 1):
            f = edges[j - 1] / np.maximum(d[m], 1e-6)              # fraction of the way to the point where the sight line crosses lip j
            ray = LIDAR_H + (-depth * k - LIDAR_H) * f
            seen[np.nonzero(m)[0]] &= ray >= -depth * (j - 1) - 0.005
    if box is not None:
        b0, b1, hw, hgt = box
        seen &= ~((x > b0) & (np.abs(y) < hw * x / b0))            # the floor behind the box
        seen &= ~((x > b0) & (x < b1) & (np.abs(y) < hw))
    if wall is not None:
        seen &= x < wall
    pts = np.stack((x, y, z), 1)[seen]
    extra = []
    if box is not None:
        m = 600
        extra.append(np.stack((np.full(m, box[0]), rng.uniform(-box[2], box[2], m), rng.uniform(0, box[3], m)), 1))
    if wall is not None:
        m = 3000
        extra.append(np.stack((np.full(m, wall), rng.uniform(-3.5, 3.5, m), rng.uniform(0, 1.2, m)), 1))
    return np.vstack([pts, *extra]).astype(np.float32)


def _look(pts, **kw):
    seen, low, obst, origin = grids_from_points(pts, extent=(-1.0, -3.5, 5.5, 3.5))
    return detect(seen, low, obst, origin, 0.1, (0.0, 0.0, 0.0), **kw)


def _selftest() -> int:
    checks: dict[str, bool] = {}

    r = _look(_scene(edge=99.0))
    checks["flat floor (noise 3 cm, like the real recording): no cliff"] = r.nearest is None and not r.cliff.any()
    r = _look(_scene(edge=99.0, density=40.0))
    checks["flat floor returned only 33% of the time (sparse lidar): still no cliff, sparse is not a hole"] = r.nearest is None
    r = _look(_scene(edge=99.0, noise=0.05))
    checks["flat floor with 5 cm noise (worse than recorded): no cliff"] = r.nearest is None

    r = _look(_scene(edge=1.8, depth=0.17))
    checks[f"a 17 cm step down 1.8 m ahead is a cliff, nearest at {r.nearest} m (the ledge is at 1.8)"] = r.nearest is not None and 1.6 < r.nearest < 2.15
    r = _look(_scene(edge=0.7, depth=0.17))
    checks[f"...and one only 0.7 m ahead, which hides less floor: found through the lower ground beyond it ({r.nearest})"] = r.nearest is not None and 0.5 <= r.nearest < 1.2
    r = _look(_scene(edge=2.0, depth=0.17))
    checks[f"...and 2.0 m ahead ({r.nearest})"] = r.nearest is not None and 1.8 < r.nearest <= 2.3
    r = _look(_scene(edge=3.4, depth=0.17))
    checks["...but not yet at 3.4 m: it only judges out to 2.5 m (it sees it on the way)"] = r.nearest is None
    r = _look(_scene(edge=2.4, depth=0.17))
    checks[f"a ledge 2.4 m ahead is right at the edge of what it judges: its shadow there is a sliver, so it is not asserted yet ({r.nearest})"] = r.nearest is None or r.nearest > 2.3
    r = _look(_scene(edge=1.5, depth=0.17, steps=4))
    checks[f"a staircase down (4 steps) starting 1.5 m ahead ({r.nearest})"] = r.nearest is not None and 1.3 < r.nearest < 1.9
    r = _look(_scene(edge=1.8, depth=0.5))
    checks[f"a 50 cm drop 1.8 m ahead: the lower ground is even more hidden, still found ({r.nearest})"] = r.nearest is not None and 1.6 < r.nearest < 2.15
    r = _look(_scene(edge=1.8, depth=0.17, angle=math.radians(40)))
    checks[f"a ledge running across at 40 degrees ({r.nearest})"] = r.nearest is not None
    r = _look(_scene(edge=1.8, depth=0.015))
    checks["a 1.5 cm dip is not a cliff"] = r.nearest is None

    r = _look(_scene(edge=99.0, box=(1.5, 1.8, 0.3, 0.5)))
    checks["the floor hidden behind a 50 cm box is a shadow, not a cliff"] = r.nearest is None
    r = _look(_scene(edge=99.0, wall=2.0))
    checks["...and so is everything beyond a wall"] = r.nearest is None
    r = _look(_scene(edge=2.0, depth=0.17, box=(1.2, 1.5, 0.3, 0.5)))
    checks[f"a ledge behind and beside a box is still found ({r.nearest})"] = r.nearest is not None and 1.8 < r.nearest < 2.3

    flat, ledge = _scene(edge=99.0), _scene(edge=1.8, depth=0.17, seed=1)
    ledge[:, 0] *= -1                                              # the same ledge, but BEHIND the dog
    behind = np.vstack((flat[flat[:, 0] >= 0], ledge[ledge[:, 0] < 0]))
    checks["a ledge BEHIND the dog is not judged (the lidar doesn't see the floor there)"] = _look(behind).nearest is None
    r = _look(np.zeros((0, 3), np.float32))
    checks["an empty message says nothing"] = r.nearest is None

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
