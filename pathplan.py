"""Corridors and paths from the Go2's lidar (numpy + scipy.ndimage: no dog needed to test it).

The dog publishes lidar points in a fixed WORLD frame plus its own pose in that frame (see obstacles.py). RoomMap drops
those points into a top-down grid: anything between z_lo and z_hi above the floor is an obstacle, floor-level returns
mean "seen, walkable", and taller-than-z_hi returns (an overhang) are ignored. plan() turns the grid into a path that
keeps the dog's radius away from obstacles and prefers the middle of a corridor, and Navigator turns that path into
move(vx, vy, yaw) commands. None of this is wired into go2.py yet. Everything the grid knows can be saved and loaded,
so a scan of a room can be reused later (the dog's world frame resets when it reboots: see the README before relying
on a loaded map).

  python pathplan.py        runs the self-test (simulated rooms, a simulated dog that lags its commands)
  python pathplan.py -v     also prints the map and the dog's track for each scenario
"""
from __future__ import annotations

import heapq
import math
import sys

import numpy as np
from scipy import ndimage

from obstacles import STAND_HEIGHT, to_dog_frame

BODY = (0.35, 0.25)          # m, half-extents of the dog's body in its own frame: lidar points inside it are the dog itself


class RoomMap:
    """Top-down obstacle grid in the world frame. Cell [iy, ix] covers x in [x0 + ix*res, x0 + (ix+1)*res), same for y."""

    def __init__(self, extent=(-8.0, -8.0, 8.0, 8.0), res: float = 0.1, z_lo: float = 0.15, z_hi: float = 1.0,
                 min_hits: float = 3.0, decay_s: float | None = 4.0):
        self.x0, self.y0 = extent[0], extent[1]
        self.res, self.z_lo, self.z_hi, self.min_hits, self.decay_s = res, z_lo, z_hi, min_hits, decay_s
        self.nx = int(math.ceil((extent[2] - extent[0]) / res))
        self.ny = int(math.ceil((extent[3] - extent[1]) / res))
        self.hits = np.zeros((self.ny, self.nx), np.float32)     # obstacle-height returns per cell (decays with time)
        self.seen = np.zeros((self.ny, self.nx), bool)           # any return at all: the cell has been looked at
        self._t: float | None = None

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor((x - self.x0) / self.res), math.floor((y - self.y0) / self.res)

    def world(self, ix: int, iy: int) -> tuple[float, float]:
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def inside(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def update(self, points_world: np.ndarray, pose: tuple[float, float, float, float], t: float | None = None) -> None:
        """Add one lidar message. pose = (x, y, z, yaw) of the dog in the same frame; t (s) drives the decay, which is
        what lets a removed obstacle fade. A static obstacle is re-hit on every message, so it never fades while seen.
        (If the dog's messages turn out to be a cumulative map rather than one scan, set decay_s=None.)"""
        x, y, z, yaw = pose
        if self.decay_s and self._t is not None and t is not None and t > self._t:
            self.hits *= math.exp(-(t - self._t) / self.decay_s)
        if t is not None:
            self._t = t
        p = np.asarray(points_world, np.float32).reshape(-1, 3)
        if len(p) == 0:
            return
        d = to_dog_frame(p, x, y, z, yaw)
        keep = ~((np.abs(d[:, 0]) < BODY[0]) & (np.abs(d[:, 1]) < BODY[1]))
        p, h = p[keep], d[keep, 2]
        ix = np.floor((p[:, 0] - self.x0) / self.res).astype(np.int64)
        iy = np.floor((p[:, 1] - self.y0) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        flat, h = iy[ok] * self.nx + ix[ok], h[ok]
        band = (h >= self.z_lo) & (h <= self.z_hi)
        self.hits += np.bincount(flat[band], minlength=self.nx * self.ny).reshape(self.ny, self.nx)
        self.seen |= np.bincount(flat[h <= self.z_hi], minlength=self.nx * self.ny).reshape(self.ny, self.nx) > 0

    def occupied(self) -> np.ndarray:
        return self.hits >= self.min_hits

    def costmap(self, pose=None, radius: float = 0.28, prefer: float = 0.8, wall_cost: float = 4.0,
                allow_unknown: bool = True, unknown_cost: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """(cost, blocked). blocked = closer than `radius` m to an obstacle (the dog's half-width plus a margin).
        cost >= 1 everywhere and rises toward `wall_cost` extra as a cell nears an obstacle, so the cheapest path runs down
        the middle of a corridor. Cells nobody has looked at cost `unknown_cost` extra, or are blocked without allow_unknown."""
        occ = self.occupied()
        dist = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, 99.0)
        blocked = dist < radius
        near = np.clip((prefer - dist) / (prefer - radius), 0.0, 1.0)
        cost = (1.0 + wall_cost * near ** 2).astype(np.float32)
        unseen = ~ndimage.binary_dilation(self.seen, iterations=max(1, round(0.3 / self.res)))   # a sparse scan leaves gaps: 0.3 m from a seen cell counts as seen
        if allow_unknown:
            cost[unseen] += unknown_cost
        else:
            blocked |= unseen
        if pose is not None:                                         # the dog stands here: its own spot can't be a wall
            iy, ix = np.ogrid[:self.ny, :self.nx]
            cx, cy = self.cell(pose[0], pose[1])
            blocked[(ix - cx) ** 2 + (iy - cy) ** 2 <= (0.5 / self.res) ** 2] = False
        return cost, blocked

    def save(self, path: str) -> None:
        np.savez_compressed(path, hits=self.hits, seen=self.seen, meta=np.array(
            [self.x0, self.y0, self.res, self.z_lo, self.z_hi, self.min_hits, self.decay_s or 0.0]))

    @classmethod
    def load(cls, path: str) -> RoomMap:
        f = np.load(path)
        x0, y0, res, z_lo, z_hi, min_hits, decay = (float(v) for v in f["meta"])
        ny, nx = f["hits"].shape
        m = cls((x0, y0, x0 + nx * res, y0 + ny * res), res, z_lo, z_hi, min_hits, decay or None)
        m.hits, m.seen = f["hits"].astype(np.float32), f["seen"].astype(bool)
        return m


_NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


def astar(cost: np.ndarray, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Cheapest 8-connected path over `cost` avoiding `blocked` (both indexed [iy, ix]); cells are (ix, iy)."""
    ny, nx = cost.shape
    if not (0 <= start[0] < nx and 0 <= start[1] < ny and 0 <= goal[0] < nx and 0 <= goal[1] < ny) or blocked[goal[1], goal[0]]:
        return None
    c, b = cost.ravel().tolist(), blocked.ravel().tolist()
    s, t = start[1] * nx + start[0], goal[1] * nx + goal[0]
    gx, gy = goal

    def h(x: int, y: int) -> float:                                    # octile distance: every step costs >= 1
        dx, dy = abs(x - gx), abs(y - gy)
        return dx + dy - 0.586 * min(dx, dy)

    g, parent, closed = [math.inf] * (nx * ny), [-1] * (nx * ny), bytearray(nx * ny)
    g[s] = 0.0
    heap = [(h(*start), s)]
    while heap:
        _, i = heapq.heappop(heap)
        if closed[i]:
            continue
        closed[i] = 1
        if i == t:
            break
        y, x = divmod(i, nx)
        for dx, dy, step in _NB:
            xx, yy = x + dx, y + dy
            if not (0 <= xx < nx and 0 <= yy < ny):
                continue
            j = yy * nx + xx
            if closed[j] or b[j] or (dx and dy and (b[y * nx + xx] or b[yy * nx + x])):   # no cutting a corner
                continue
            ng = g[i] + step * c[j]
            if ng < g[j]:
                g[j], parent[j] = ng, i
                heapq.heappush(heap, (ng + h(xx, yy), j))
    if not closed[t]:
        return None
    out, i = [], t
    while i != -1:
        out.append((i % nx, i // nx))
        i = parent[i]
    return out[::-1]


def _line_ok(cost, blocked, a, b, soft) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    for k in range(n + 1):
        x = round(a[0] + (b[0] - a[0]) * k / n)
        y = round(a[1] + (b[1] - a[1]) * k / n)
        if blocked[y, x] or cost[y, x] > 1.0 + soft:
            return False
    return True


def shortcut(path: list[tuple[int, int]], cost: np.ndarray, blocked: np.ndarray, soft: float = 1.0) -> list[tuple[int, int]]:
    """Drop waypoints that a straight line can skip, but only over cheap (well away from obstacles) cells, so the
    smoothed path still keeps to the middle of a corridor instead of cutting along its wall."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_ok(cost, blocked, path[i], path[j], soft):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan(room: RoomMap, pose, goal: tuple[float, float], snap: float = 0.6, **cost_kw) -> list[tuple[float, float]] | None:
    """Path (world x, y waypoints, the dog first) from the dog to `goal`, or None when there isn't one. A goal that
    sits inside the safety margin is moved to the nearest free cell within `snap` m."""
    cost, blocked = room.costmap(pose, **cost_kw)
    s, g = room.cell(pose[0], pose[1]), room.cell(*goal)
    if not (room.inside(*s) and room.inside(*g)):
        return None
    if blocked[g[1], g[0]]:
        _, idx = ndimage.distance_transform_edt(blocked, return_indices=True)
        g = (int(idx[1][g[1], g[0]]), int(idx[0][g[1], g[0]]))
        if math.hypot(*(a - b for a, b in zip(room.world(*g), goal))) > snap:
            return None
    cells = astar(cost, blocked, s, g)
    if cells is None:
        return None
    return [(pose[0], pose[1])] + [room.world(*c) for c in shortcut(cells, cost, blocked)[1:]]


def steer(path: list[tuple[float, float]], pose, vmax: float = 0.5, wmax: float = 1.0, lookahead: float = 0.6,
          goal_tol: float = 0.3) -> tuple[float, float, float] | None:
    """Pure pursuit: aim at the point `lookahead` m further along the path. Returns (vx, vy, yaw rate) for
    robot.move(), or None once within goal_tol of the path's end. Turns in place when the target is well off to one
    side, and slows as it nears the goal."""
    x, y, _, yaw = pose
    pts = np.asarray(path, float)
    left = math.hypot(*(pts[-1] - (x, y)))
    if left <= goal_tol:
        return None
    seg = np.hypot(*(np.diff(pts, axis=0).T)) if len(pts) > 1 else np.zeros(0)
    near = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
    target = pts[-1]
    walked = 0.0
    for k in range(near, len(pts) - 1):                              # walk `lookahead` m along the path from the nearest point
        if walked + seg[k] >= lookahead:
            target = pts[k] + (pts[k + 1] - pts[k]) * (lookahead - walked) / seg[k]
            break
        walked += seg[k]
    err = math.atan2(target[1] - y, target[0] - x) - yaw
    err = math.atan2(math.sin(err), math.cos(err))
    vx = min(vmax, 0.2 + left) * max(0.0, math.cos(err)) ** 2
    return vx, 0.0, max(-wmax, min(wmax, 2.0 * err))


class Navigator:
    """Call step() with every lidar message's pose after RoomMap.update(); it replans a couple of times a second and returns
    the command to send. .state is "going", "arrived" or "no_path" (the dog stands still until a path appears)."""

    def __init__(self, room: RoomMap, goal: tuple[float, float], replan_every: float = 0.5, goal_tol: float = 0.3,
                 vmax: float = 0.5, wmax: float = 1.0, **cost_kw):
        self.room, self.goal, self.replan_every, self.goal_tol = room, goal, replan_every, goal_tol
        self.vmax, self.wmax, self.cost_kw = vmax, wmax, cost_kw
        self.path: list[tuple[float, float]] | None = None
        self.state = "going"
        self._planned = -math.inf

    def step(self, pose, now: float) -> tuple[float, float, float]:
        if math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1]) <= self.goal_tol:
            self.state = "arrived"
            return 0.0, 0.0, 0.0
        if self.path is None or now - self._planned >= self.replan_every:
            self.path, self._planned = plan(self.room, pose, self.goal, **self.cost_kw), now
        if self.path is None:
            self.state = "no_path"
            return 0.0, 0.0, 0.0
        self.state = "going"
        return steer(self.path, pose, self.vmax, self.wmax, goal_tol=self.goal_tol) or (0.0, 0.0, 0.0)


def ascii_map(room: RoomMap, pose=None, path=None, goal=None, radius: float = 0.28, cols: int = 78) -> str:
    """Terminal picture of the grid: # obstacle, + inside the safety margin, . free, blank never seen, * path, D dog, G goal."""
    _, blocked = room.costmap(pose, radius=radius)
    occ = room.occupied()
    k = max(1, math.ceil(room.nx / cols))
    grid = [[" " if not room.seen[y, x] else "#" if occ[y, x] else "+" if blocked[y, x] else "." for x in range(0, room.nx, k)]
            for y in range(0, room.ny, k)]

    def put(xy, ch):
        ix, iy = room.cell(*xy)
        if room.inside(ix, iy):
            grid[iy // k][ix // k] = ch
    for xy in path or []:
        put(xy, "*")
    if goal:
        put(goal, "G")
    if pose:
        put(pose[:2], "D")
    return "\n".join("".join(r) for r in grid[::-1])


# ---- self-test: simulated rooms ----------------------------------------------------------------------------
class Sim:
    """A room of upright boxes (x0, y0, x1, y1, height) with a floor, seen by a lidar that returns a random third of the
    points within 6 m each tick (no occlusion: it sees through walls, which is kinder than the real thing), and a dog that
    lags its commands (first-order, 0.25 s) like the real one."""

    def __init__(self, boxes, start=(0.0, 0.0), yaw=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.boxes = list(boxes)
        self.x, self.y, self.yaw = start[0], start[1], yaw
        self.v = np.zeros(3)
        self.t = 0.0
        self.floor = np.stack((self.rng.uniform(-7, 7, 9000), self.rng.uniform(-7, 7, 9000), self.rng.normal(0, 0.01, 9000)), 1)
        self.cloud = np.vstack((self.floor, *(self._points(b) for b in self.boxes))).astype(np.float32)
        self.track = [(self.x, self.y)]
        self.segs: list = []                                    # slanted walls: (a, b, height, thickness)

    def _points(self, b):
        n = max(40, int((b[2] - b[0]) * (b[3] - b[1]) * b[4] * 2500))
        return np.stack((self.rng.uniform(b[0], b[2], n), self.rng.uniform(b[1], b[3], n), self.rng.uniform(0, b[4], n)), 1)

    def add(self, b):
        self.boxes.append(b)
        self.cloud = np.vstack((self.cloud, self._points(b))).astype(np.float32)

    def add_segment(self, a, b, h=1.8, thick=0.1):
        """A wall of any angle from point a to point b."""
        n = max(40, int(math.dist(a, b) * thick * h * 2500) + int(math.dist(a, b) * h * 300))
        t, off = self.rng.random(n), (self.rng.random(n) - 0.5) * thick
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        px = a[0] + (b[0] - a[0]) * t - math.sin(ang) * off
        py = a[1] + (b[1] - a[1]) * t + math.cos(ang) * off
        self.segs.append((a, b, h, thick))
        self.cloud = np.vstack((self.cloud, np.stack((px, py, self.rng.uniform(0, h, n)), 1))).astype(np.float32)

    def remove_last(self):
        b = self.boxes.pop()
        m = ~((self.cloud[:, 0] >= b[0]) & (self.cloud[:, 0] <= b[2]) & (self.cloud[:, 1] >= b[1]) & (self.cloud[:, 1] <= b[3]) & (self.cloud[:, 2] > 0.05))
        self.cloud = self.cloud[m]

    def pose(self):
        return self.x, self.y, STAND_HEIGHT, self.yaw

    def scan(self):
        c = self.cloud[self.rng.random(len(self.cloud)) < 0.33]
        c = c[np.hypot(c[:, 0] - self.x, c[:, 1] - self.y) < 6.0]
        return c + self.rng.normal(0, 0.01, c.shape).astype(np.float32)

    def tick(self, cmd, dt=0.1):
        self.v += (np.asarray(cmd) - self.v) * (1 - math.exp(-dt / 0.25))
        self.yaw += self.v[2] * dt
        self.x += (self.v[0] * math.cos(self.yaw) - self.v[1] * math.sin(self.yaw)) * dt
        self.y += (self.v[0] * math.sin(self.yaw) + self.v[1] * math.cos(self.yaw)) * dt
        self.t += dt
        self.track.append((self.x, self.y))

    def gap(self):
        """Smallest distance (m) from the dog's track to any obstacle above knee height."""
        best = 9.0
        for x, y in self.track:
            for b in self.boxes:
                if b[4] > 0.2:
                    best = min(best, math.hypot(max(b[0] - x, 0, x - b[2]), max(b[1] - y, 0, y - b[3])))
            for a, b, h, thick in self.segs:
                ab = (b[0] - a[0], b[1] - a[1])
                u = max(0.0, min(1.0, ((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)))
                best = min(best, math.hypot(x - a[0] - u * ab[0], y - a[1] - u * ab[1]) - thick / 2)
        return best


def run(sim: Sim, goal, seconds=60.0, events=(), verbose=False, **nav_kw):
    """Drive the sim to `goal`. events = [(t, fn(sim))] fire once when the clock passes t. Returns (navigator, room)."""
    room = RoomMap(extent=(-7, -7, 7, 7), decay_s=nav_kw.pop("decay_s", 4.0))
    nav = Navigator(room, goal, **nav_kw)
    pending = sorted(events, key=lambda e: e[0])
    while sim.t < seconds:
        while pending and sim.t >= pending[0][0]:
            pending.pop(0)[1](sim)
        room.update(sim.scan(), sim.pose(), sim.t)
        cmd = nav.step(sim.pose(), sim.t)
        if nav.state == "arrived":
            break
        sim.tick(cmd)
    if verbose:
        print(ascii_map(room, sim.pose(), nav.path, goal))
        print(f"    t={sim.t:.1f}s state={nav.state} at ({sim.x:.2f}, {sim.y:.2f}) closest approach {sim.gap():.2f} m")
    return nav, room


def _selftest(verbose: bool) -> int:
    checks: dict[str, bool] = {}
    walls = [(-6, -4, 6, -3.9, 1.8), (-6, 3.9, 6, 4, 1.8), (-6, -4, -5.9, 4, 1.8), (5.9, -4, 6, 4, 1.8)]

    # -- the grid itself
    rng = np.random.default_rng(3)
    floor = np.stack((rng.uniform(-3, 3, 4000), rng.uniform(-3, 3, 4000), rng.normal(0, 0.01, 4000)), 1)
    pose = (0.0, 0.0, STAND_HEIGHT, 0.0)

    def grid_with(cloud, **kw):
        m = RoomMap(extent=(-4, -4, 4, 4), **kw)
        m.update(np.vstack((floor, cloud)).astype(np.float32), pose, 0.0)
        return m
    box = lambda x, y, w, h, n=300: np.stack((x + (rng.random(n) - .5) * w, y + (rng.random(n) - .5) * w, rng.random(n) * h), 1)  # noqa: E731
    checks["floor alone is walkable, not an obstacle"] = not grid_with(np.zeros((0, 3))).occupied().any()
    m = grid_with(box(1.5, 0.0, 0.3, 0.6))
    checks["a box shows up as obstacle cells at its own position"] = bool(m.occupied()[m.cell(1.5, 0.0)[1], m.cell(1.5, 0.0)[0]])
    checks["a 5 cm ridge is walkable"] = not grid_with(box(1.0, 0.0, 0.5, 0.05)).occupied().any()
    checks["a 12 cm ridge is walkable, a 20 cm kerb is not"] = (
        not grid_with(box(1.0, 0.0, 0.5, 0.12)).occupied().any() and grid_with(np.stack((np.full(200, 1.0), np.full(200, 0.0), np.full(200, 0.2)), 1)).occupied().any())
    checks["an overhang (table top at 1.2 m) does not block"] = not grid_with(np.stack((rng.uniform(.8, 1.4, 200), rng.uniform(-.3, .3, 200), np.full(200, 1.2)), 1)).occupied().any()
    checks["two stray points are noise"] = not grid_with(np.array([[1.0, 0.0, 0.5], [1.0, 0.02, 0.4]])).occupied().any()
    checks["the dog's own body is not an obstacle"] = not grid_with(box(0.0, 0.0, 0.3, 0.5, 200)).occupied().any()
    m = RoomMap(extent=(-4, -4, 4, 4), decay_s=2.0)
    m.update(np.vstack((floor, box(1.5, 0.0, 0.3, 0.6))).astype(np.float32), pose, 0.0)
    m.update(floor.astype(np.float32), pose, 0.5)
    early = m.occupied().any()
    m.update(floor.astype(np.float32), pose, 12.0)
    checks["a removed obstacle fades once it stops being seen (not at once, but within a few seconds)"] = early and not m.occupied().any()
    m = grid_with(box(1.5, 0.5, 0.3, 0.6))
    m.save("_room_test.npz")
    m2 = RoomMap.load("_room_test.npz")
    import os
    os.remove("_room_test.npz")
    checks["a saved map loads back identical"] = bool(np.array_equal(m.hits, m2.hits) and np.array_equal(m.seen, m2.seen) and m2.res == m.res and m2.x0 == m.x0)
    m = RoomMap(extent=(-4, -4, 4, 4))
    m.update(floor[floor[:, 0] < 0.5].astype(np.float32), pose, 0.0)                    # only the left half was ever looked at
    checks["without allow_unknown a goal in unseen space has no path"] = plan(m, pose, (2.0, 0.0), allow_unknown=False) is None
    checks["with allow_unknown the same goal is reachable"] = plan(m, pose, (2.0, 0.0)) is not None

    # -- A* and the smoothing
    cost = np.ones((20, 20), np.float32)
    blocked = np.zeros((20, 20), bool)
    blocked[5:15, 10] = True
    p = astar(cost, blocked, (2, 10), (18, 10))
    checks["A* goes around a wall"] = p is not None and p[0] == (2, 10) and p[-1] == (18, 10) and not any(blocked[y, x] for x, y in p)
    blocked[:, 10] = True
    checks["A* returns None when the wall is complete"] = astar(cost, blocked, (2, 10), (18, 10)) is None

    # -- driving in simulated rooms
    nav, _ = run(sim := Sim(walls), (4.0, 0.0), verbose=verbose)
    checks[f"open floor: arrives at the goal in a straight-ish line ({sim.t:.1f} s)"] = nav.state == "arrived" and sim.t < 20 and max(abs(y) for _, y in sim.track) < 0.5

    nav, _ = run(sim := Sim(walls + [(1.6, -0.3, 2.2, 0.3, 0.6)]), (4.0, 0.0), verbose=verbose)
    checks[f"a box in the way: goes round it and arrives, never closer than 0.18 m ({sim.gap():.2f} m)"] = nav.state == "arrived" and sim.gap() > 0.18

    door = walls + [(2.95, -4, 3.05, 0.5, 1.8), (2.95, 1.5, 3.05, 4, 1.8)]           # a wall across the room, 1.0 m doorway at y in [0.5, 1.5]
    nav, _ = run(sim := Sim(door), (5.0, -1.0), verbose=verbose)
    cross = [y for (x, y), (x2, _) in zip(sim.track, sim.track[1:]) if x < 3.0 <= x2]
    checks[f"a doorway: goes through it (crossed the wall at y={cross[0]:.2f}) and arrives" if cross else "a doorway: never crossed the wall"] = (
        nav.state == "arrived" and bool(cross) and 0.5 < cross[0] < 1.5 and sim.gap() > 0.18)

    slit = walls + [(2.95, -4, 3.05, 0.6, 1.8), (2.95, 1.0, 3.05, 4, 1.8)]           # only 0.4 m gap: too narrow for the dog
    nav, _ = run(sim := Sim(slit), (5.0, 0.0), seconds=12.0, verbose=verbose)
    checks[f"a 0.4 m gap is refused: state {nav.state}, the dog stays on its side (x={sim.x:.2f})"] = nav.state == "no_path" and sim.x < 2.9

    corridor = walls + [(1, 0.8, 7, 0.9, 1.8), (1, -0.9, 7, -0.8, 1.8)]              # a 1.6 m wide corridor
    nav, _ = run(sim := Sim(corridor, start=(0.0, 0.5)), (5.5, 0.0), verbose=verbose)
    inside = [abs(y) for x, y in sim.track if 2.5 < x < 5.5]
    checks[f"a corridor: it settles on the middle (median offset {np.median(inside):.2f} m), no wall touched"] = (
        nav.state == "arrived" and len(inside) > 5 and float(np.median(inside)) < 0.25 and sim.gap() > 0.18)

    late = (6.0, lambda s: s.add((3.6, -0.4, 4.2, 0.4, 0.8)))                        # somebody puts a box in the way, about a metre ahead, after it has set off
    nav, _ = run(sim := Sim(walls), (5.0, 0.0), events=[late], verbose=verbose)
    checks[f"a box appears mid-walk: it replans round it and arrives ({sim.gap():.2f} m clear)"] = nav.state == "arrived" and sim.gap() > 0.15

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest("-v" in sys.argv))
