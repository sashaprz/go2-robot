"""Corridors and paths from the Go2's lidar (numpy + scipy.ndimage: no dog needed to test it).

The dog publishes lidar points in a fixed WORLD frame plus its own pose in that frame (see obstacles.py). RoomMap drops
those points into a top-down grid: anything between z_lo and HEAD_TOP (the top of what this lidar returns, ~1.2 m) above the floor
is an obstacle, and anything from 0.45 m up is also kept as an OVERHEAD layer that gets a person-wide berth (a table top the dog
walks under still hits the person following it). Floor-level returns mean "seen, walkable". With cliffs=True it also finds drop-offs
(dropoff.py: stairs down, ledges) and remembers them as no-go. plan() turns the grid into a path that
keeps the dog's radius away from obstacles and prefers the middle of a corridor, and Navigator turns that path into
move(vx, vy, yaw) commands. guide.py runs it on the real dog (standalone); none of it is wired into go2.py yet. Everything the grid knows can be saved and loaded,
so a scan of a room can be reused later (the dog's world frame resets when it reboots: see the README before relying
on a loaded map).

  python pathplan.py        runs the self-test (simulated rooms, a simulated dog that lags its commands)
  python pathplan.py -v     also prints the map and the dog's track for each scenario
"""
from __future__ import annotations

import heapq
import math
import os
import sys

import numpy as np
from scipy import ndimage

import dropoff
from obstacles import HEAD_TOP, STAND_HEIGHT, to_dog_frame

BODY = (0.35, 0.25)          # m, half-extents of the dog's body in its own frame: lidar points inside it are the dog itself


class RoomMap:
    """Top-down obstacle grid in the world frame. Cell [iy, ix] covers x in [x0 + ix*res, x0 + (ix+1)*res), same for y."""

    def __init__(self, extent=(-8.0, -8.0, 8.0, 8.0), res: float = 0.1, z_lo: float = 0.15, z_hi: float = HEAD_TOP,
                 min_hits: float = 3.0, decay_s: float | None = 4.0, replace: bool = False, hold_radius: float = 0.0,
                 z_head: float = 0.45, cliffs: bool = False, cliff_far: float = 2.5, cliff_forget: int = 12):
        """replace=True: every lidar message is already the dog's own persistent map (recordings of the real dog: 91-99% of the
        points repeat from one message to the next), so the grid is rebuilt from each message instead of accumulating them and
        decay_s is ignored. hold_radius (m): the real lidar returns nothing standing up nearer than about 1 m, so cells that
        close to the dog keep what was seen before it got that near (a new return there still counts; only "gone" is not believed).
        z_hi is where the lidar STOPS SEEING (~1.2 m), not where the world is clear: a hanging sign or a low ceiling above it is not in
        the map. z_head: returns from this height up also go in an overhead layer (costmap gives it handler_radius, a person's width).
        cliffs=True: look for drop-offs with dropoff.detect (out to cliff_far m ahead) and remember them in .cliff. A flagged cell is only
        forgotten after cliff_forget messages in a row that judged it and did not find it again (~1.5 s): a ledge seen at an angle, or with a
        ragged shadow, flickers, and a flicker must not open a path to the edge."""
        self.x0, self.y0 = extent[0], extent[1]
        self.res, self.z_lo, self.z_hi, self.min_hits, self.decay_s = res, z_lo, z_hi, min_hits, decay_s
        self.replace, self.hold_radius, self.z_head, self.cliffs, self.cliff_far = replace, hold_radius, z_head, cliffs, cliff_far
        self.cliff_forget = cliff_forget
        self.nx = int(math.ceil((extent[2] - extent[0]) / res))
        self.ny = int(math.ceil((extent[3] - extent[1]) / res))
        self._xc = self.x0 + (np.arange(self.nx) + 0.5) * res             # cell centres, for the hold radius
        self._yc = self.y0 + (np.arange(self.ny) + 0.5) * res
        self.hits = np.zeros((self.ny, self.nx), np.float32)     # obstacle-height returns per cell (decays with time)
        self.hits_hi = np.zeros((self.ny, self.nx), np.float32)  # ...of those, the ones from z_head up (the person's height)
        self.low = np.zeros((self.ny, self.nx), np.float32)      # returns well BELOW the floor: the ground under a ledge
        self.seen = np.zeros((self.ny, self.nx), bool)           # any return at all: the cell has been looked at
        self.cliff = np.zeros((self.ny, self.nx), bool)          # a drop-off: remembered until the dog looks again and it isn't (for a while)
        self._cliff_miss = np.zeros((self.ny, self.nx), np.uint8)   # messages in a row that judged a flagged cell and did not find it
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
        near = None                                                  # cells too close to the dog for the lidar to see into
        if self.hold_radius > 0:
            near = np.hypot(self._xc[None, :] - x, self._yc[:, None] - y) < self.hold_radius
        if not self.replace and self.decay_s and self._t is not None and t is not None and t > self._t:
            f = math.exp(-(t - self._t) / self.decay_s)
            for name in ("hits", "hits_hi", "low"):
                g = getattr(self, name)
                setattr(self, name, g * f if near is None else np.where(near, g, g * f).astype(np.float32))
        if t is not None:
            self._t = t
        p = np.asarray(points_world, np.float32).reshape(-1, 3)
        if len(p) == 0:
            if self.replace:                                         # an empty map: nothing is seen, so only the held cells survive
                for name in ("hits", "hits_hi", "low"):
                    g = getattr(self, name)
                    setattr(self, name, g * 0 if near is None else np.where(near, g, 0.0).astype(np.float32))
            return
        d = to_dog_frame(p, x, y, z, yaw)
        keep = ~((np.abs(d[:, 0]) < BODY[0]) & (np.abs(d[:, 1]) < BODY[1]))
        p, h = p[keep], d[keep, 2]
        ix = np.floor((p[:, 0] - self.x0) / self.res).astype(np.int64)
        iy = np.floor((p[:, 1] - self.y0) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        flat, h = iy[ok] * self.nx + ix[ok], h[ok]

        def count(mask):
            return np.bincount(flat[mask], minlength=self.nx * self.ny).reshape(self.ny, self.nx).astype(np.float32)

        def merge(old, new):
            if self.replace:
                return new if near is None else np.where(near, np.maximum(old, new), new)
            return old + new
        self.hits = merge(self.hits, count((h >= self.z_lo) & (h <= self.z_hi)))
        self.hits_hi = merge(self.hits_hi, count((h >= self.z_head) & (h <= self.z_hi)))
        self.low = merge(self.low, count((h < dropoff.DROP_Z) & (h > -1.5)))
        self.seen |= count(h <= self.z_hi) > 0
        if self.cliffs:
            self._find_cliffs(pose)

    def _find_cliffs(self, pose) -> None:
        far = self.cliff_far + 0.3
        ix0, iy0 = self.cell(pose[0] - far, pose[1] - far)
        ix1, iy1 = self.cell(pose[0] + far, pose[1] + far)
        ix0, iy0, ix1, iy1 = max(ix0, 0), max(iy0, 0), min(ix1, self.nx - 1), min(iy1, self.ny - 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        sl = (slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))
        r = dropoff.detect(self.seen[sl], self.low[sl], self.occupied()[sl], (self.x0 + ix0 * self.res, self.y0 + iy0 * self.res),
                           self.res, (pose[0], pose[1], pose[3]), far=self.cliff_far)
        found, flagged = r.window & r.cliff, self.cliff[sl]
        missed = r.window & ~r.cliff & flagged                              # judged again, and not found this time
        miss = np.where(found, 0, np.where(missed, np.minimum(self._cliff_miss[sl], 250) + 1, self._cliff_miss[sl])).astype(np.uint8)
        self._cliff_miss[sl] = miss
        self.cliff[sl] = (flagged | found) & ~(missed & (miss >= self.cliff_forget))   # found: flagged at once. Not found: forgotten only after a run of misses

    def occupied(self) -> np.ndarray:
        return self.hits >= self.min_hits

    def occupied_high(self) -> np.ndarray:
        return self.hits_hi >= self.min_hits

    def nearest_ahead(self, mask: np.ndarray, pose, max_r: float, half_width: float | None = None) -> float | None:
        """Distance (m) to the nearest True cell of `mask` in the half-plane ahead of the dog, within max_r (and, if given, within
        half_width m of its line of travel). None if there is none."""
        iy, ix = np.nonzero(mask)
        if len(ix) == 0:
            return None
        dx, dy = self._xc[ix] - pose[0], self._yc[iy] - pose[1]
        c, s = math.cos(pose[3]), math.sin(pose[3])
        u, v = dx * c + dy * s, -dx * s + dy * c
        m = (u > 0.0) & (np.hypot(u, v) <= max_r)
        if half_width is not None:
            m &= np.abs(v) <= half_width
        return float(np.hypot(u[m], v[m]).min()) if m.any() else None

    def nearest_any(self, mask: np.ndarray, pose, max_r: float) -> float | None:
        """Distance (m) to the nearest True cell of `mask` in ANY direction from the dog, within max_r; None if there is none."""
        iy, ix = np.nonzero(mask)
        if len(ix) == 0:
            return None
        d = np.hypot(self._xc[ix] - pose[0], self._yc[iy] - pose[1])
        d = d[d <= max_r]
        return float(d.min()) if d.size else None

    def cliff_ahead(self, pose, dist: float, half_width: float = 0.3) -> float | None:
        """Distance ahead to the nearest drop-off inside a strip `half_width` m either side of where the dog is heading, or None."""
        return self.nearest_ahead(self.cliff, pose, dist, half_width) if self.cliff.any() else None

    def free_behind(self, pose, dist: float, half_width: float = 0.3) -> bool:
        """Is the strip `dist` m behind the dog, `half_width` either side, free of obstacles and drop-offs? (Only what the map knows.)"""
        iy, ix = np.nonzero(self.occupied() | self.occupied_high() | self.cliff)
        if len(ix) == 0:
            return True
        dx, dy = self._xc[ix] - pose[0], self._yc[iy] - pose[1]
        c, s = math.cos(pose[3]), math.sin(pose[3])
        u, v = dx * c + dy * s, -dx * s + dy * c
        return not bool(((u < 0.0) & (u > -dist) & (np.abs(v) <= half_width)).any())

    def costmap(self, pose=None, radius: float = 0.28, prefer: float = 0.8, wall_cost: float = 4.0,
                allow_unknown: bool = True, unknown_cost: float = 2.0, handler_radius: float = 0.35,
                cliff_radius: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
        """(cost, blocked). blocked = closer than `radius` m to an obstacle (the dog's half-width plus a margin), closer than
        `handler_radius` to anything that reaches person height (a person is wider than the dog and its hips and shoulders are what a
        table edge or a counter hits), or closer than `cliff_radius` to a drop-off (the first cliff cell is about 0.1 m past the lip, and the dog's nose is 0.35 m ahead of its
        centre, so 0.75 m keeps the dog's centre ~0.6 m and its nose ~0.25 m from the edge). cost >= 1 everywhere and rises toward `wall_cost`
        extra as a cell nears an obstacle, so the cheapest path runs down the middle of a corridor. Cells nobody has looked at cost
        `unknown_cost` extra, or are blocked without allow_unknown."""
        occ = self.occupied()
        dist = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, 99.0)
        blocked = dist < radius
        hi = self.occupied_high()
        if hi.any():
            blocked |= ndimage.distance_transform_edt(~hi) * self.res < handler_radius
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
        if self.cliff.any():                                         # ...but a drop-off stays a drop-off wherever the dog stands
            dc = ndimage.distance_transform_edt(~self.cliff) * self.res
            blocked |= dc < cliff_radius
            cost += (wall_cost * np.clip((cliff_radius + 0.5 - dc) / 0.5, 0.0, 1.0) ** 2).astype(np.float32)
        return cost, blocked

    def save(self, path: str) -> None:
        np.savez_compressed(path, hits=self.hits, hits_hi=self.hits_hi, low=self.low, cliff=self.cliff, seen=self.seen, meta=np.array(
            [self.x0, self.y0, self.res, self.z_lo, self.z_hi, self.min_hits, self.decay_s or 0.0, float(self.replace), self.hold_radius,
             self.z_head, float(self.cliffs)]))

    @classmethod
    def load(cls, path: str) -> RoomMap:
        f = np.load(path)
        meta = [float(v) for v in f["meta"]]
        meta += [0.0] * (9 - len(meta))                              # maps saved before replace / hold_radius existed lack those two
        x0, y0, res, z_lo, z_hi, min_hits, decay, replace, hold = meta[:9]
        z_head, cliffs = (meta[9], meta[10]) if len(meta) >= 11 else (0.45, 0.0)
        ny, nx = f["hits"].shape
        m = cls((x0, y0, x0 + nx * res, y0 + ny * res), res, z_lo, z_hi, min_hits, decay or None, bool(replace), hold, z_head, bool(cliffs))
        m.hits, m.seen = f["hits"].astype(np.float32), f["seen"].astype(bool)
        for name in ("hits_hi", "low"):
            if name in f.files:
                setattr(m, name, f[name].astype(np.float32))
        if "cliff" in f.files:
            m.cliff = f["cliff"].astype(bool)
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
    """Terminal picture of the grid: # obstacle, + inside the safety margin, v drop-off, . free, blank never seen, * path, D dog, G goal."""
    _, blocked = room.costmap(pose, radius=radius)
    occ, cliff = room.occupied(), room.cliff
    k = max(1, math.ceil(room.nx / cols))
    grid = [["v" if cliff[y, x] else " " if not room.seen[y, x] else "#" if occ[y, x] else "+" if blocked[y, x] else "." for x in range(0, room.nx, k)]
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

    def __init__(self, boxes, start=(0.0, 0.0), yaw=0.0, seed=0, blind: float = 0.0, pits=(), floor_per_m2: float | None = None,
                 floor_half: float = 6.5, noise: float = 0.01):
        """pits = [(x0, y0, x1, y1, depth)]: in that rectangle the floor is `depth` m lower (later entries win, so a staircase down is a few
        overlapping rectangles of growing depth); the lidar cannot see it past the lip, as on the real dog. floor_per_m2: floor returns per m^2
        (the default, 9000 over +-7 m, is only ~46/m^2, far sparser than the real dog: ~100% of 0.1 m cells out to 1.5 m). noise: floor z noise, m."""
        self.blind = blind                                      # nothing standing up nearer than this to the dog is ever returned (the real lidar: ~1 m)
        self.pits, self.noise = list(pits), noise
        self.rng = np.random.default_rng(seed)
        self.boxes = list(boxes)
        self.x, self.y, self.yaw = start[0], start[1], yaw
        self.v = np.zeros(3)
        self.t = 0.0
        n, half = (int(floor_per_m2 * (2 * floor_half) ** 2), floor_half) if floor_per_m2 else (9000, 7.0)
        fx, fy = self.rng.uniform(-half, half, n), self.rng.uniform(-half, half, n)
        self.floor = np.stack((fx, fy, self._terrain(fx, fy) + self.rng.normal(0, noise, n)), 1)
        self.cloud = np.vstack((self.floor, *(self._points(b) for b in self.boxes))).astype(np.float32)
        self.track = [(self.x, self.y)]
        self.segs: list = []                                    # slanted walls: (a, b, height, thickness)

    def _points(self, b):
        n = max(40, int((b[2] - b[0]) * (b[3] - b[1]) * b[4] * 2500))
        return np.stack((self.rng.uniform(b[0], b[2], n), self.rng.uniform(b[1], b[3], n), self.rng.uniform(0, b[4], n)), 1)

    def _terrain(self, x, y):
        """Floor height at (x, y): 0, or minus the depth of the pit it is in."""
        h = np.zeros(np.shape(x))
        for x0, y0, x1, y1, depth in self.pits:
            h = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), -depth, h)
        return h

    def _hidden(self, c):
        """True where the sight line from the lidar (0.35 m up) to a point below floor level is cut by the lip of a pit."""
        hid = np.zeros(len(c), bool)
        if not self.pits or len(c) == 0:
            return hid
        idx = np.nonzero(c[:, 2] < -0.03)[0]
        if len(idx) == 0:
            return hid
        q = c[idx]
        f = np.linspace(0.0, 1.0, 41)[1:-1]                     # fractions of the way from the lidar to the point
        xs, ys = self.x + (q[:, 0] - self.x)[:, None] * f, self.y + (q[:, 1] - self.y)[:, None] * f
        ray = 0.35 + (q[:, 2] - 0.35)[:, None] * f
        hid[idx] = (ray < self._terrain(xs, ys) - 0.005).any(1)
        return hid

    def pit_gap(self):
        """Closest the dog's track came to a pit, m (negative: it went in)."""
        best = 9.0
        for x, y in self.track:
            for x0, y0, x1, y1, _ in self.pits:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    best = min(best, -min(x - x0, x1 - x, y - y0, y1 - y))
                else:
                    best = min(best, math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)))
        return best

    def add(self, b):
        self.boxes.append(b)
        self.cloud = np.vstack((self.cloud, self._points(b))).astype(np.float32)

    def add_slab(self, x0, y0, x1, y1, z0, z1):
        """Something that floats: a table top or shelf from height z0 to z1 with nothing under it."""
        self.boxes.append((x0, y0, x1, y1, z1))
        n = max(60, int((x1 - x0) * (y1 - y0) * 400))
        pts = np.stack((self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n), self.rng.uniform(z0, z1, n)), 1)
        self.cloud = np.vstack((self.cloud, pts)).astype(np.float32)

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
        d = np.hypot(c[:, 0] - self.x, c[:, 1] - self.y)
        c = c[(d < 6.0) & ~((c[:, 2] > 0.1) & (d < self.blind))]
        c = c[~self._hidden(c)]
        return c + self.rng.normal(0, self.noise, c.shape).astype(np.float32)

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


class PersistentSim(Sim):
    """The real dog's lidar as the recordings showed it: each message is a persistent MAP (everything seen so far, not one scan),
    nothing standing up is returned nearer than `blind` m, and an object that leaves lingers `ghost_s` s. Like Sim, the dog lags
    its commands and the lidar sees through walls (but not over the lip of a pit). The floor is dense (`floor_per_m2`, default 250)."""

    def __init__(self, boxes, start=(0.0, 0.0), yaw=0.0, seed=0, blind: float = 1.0, ghost_s: float = 40.0, forgets_blind: bool = False,
                 floor_per_m2: float | None = 250.0, **kw):
        super().__init__(boxes, start, yaw, seed, blind=blind, floor_per_m2=floor_per_m2, **kw)
        self.ghost_s = ghost_s
        self.forgets_blind = forgets_blind                  # the recordings can't say whether the dog's map keeps what it saw before it got within ~1 m: True = it doesn't
        self.alive = np.ones(len(self.cloud), bool)
        self.last = np.full(len(self.cloud), -1e9)          # when each point was last returned by the lidar
        self.jit = self.rng.normal(0, self.noise, self.cloud.shape).astype(np.float32)      # a map's points don't jitter from message to message

    def _sync(self):
        n = len(self.cloud) - len(self.alive)
        if n > 0:
            self.alive = np.concatenate((self.alive, np.ones(n, bool)))
            self.last = np.concatenate((self.last, np.full(n, -1e9)))
            self.jit = np.concatenate((self.jit, self.rng.normal(0, self.noise, (n, 3)).astype(np.float32)))

    def remove_last(self):
        b = self.boxes.pop()
        self.alive &= ~((self.cloud[:, 0] >= b[0]) & (self.cloud[:, 0] <= b[2]) & (self.cloud[:, 1] >= b[1]) & (self.cloud[:, 1] <= b[3]) & (self.cloud[:, 2] > 0.05))

    def scan(self):
        self._sync()
        c = self.cloud
        d = np.hypot(c[:, 0] - self.x, c[:, 1] - self.y)
        zone = (d < 6.0) & ~((c[:, 2] > 0.1) & (d < self.blind))                 # where the lidar can see
        cand = np.nonzero(self.alive & zone & (self.rng.random(len(c)) < 0.33))[0]
        cand = cand[~self._hidden(c[cand])]                                       # the lip of a pit hides the ground past it
        self.last[cand] = self.t
        self.last[zone & (self.last > -1e8) & (self.t - self.last > self.ghost_s)] = -1e9   # should have been seen again and wasn't: forgotten for good
        keep = self.last > -1e8
        if self.forgets_blind:
            keep &= ~((c[:, 2] > 0.1) & (d < self.blind))
        return c[keep] + self.jit[keep]


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
    slab = lambda z, x=1.5, w=0.3, n=300: np.stack((x + (rng.random(n) - .5) * w, (rng.random(n) - .5) * w, np.full(n, z)), 1)  # noqa: E731
    checks["a table top at 0.8 m (the dog fits under it, the person behind it does not) blocks"] = grid_with(slab(0.8)).occupied().any()
    checks["a counter edge at 1.1 m, above the old 1.0 m limit, blocks too"] = grid_with(slab(1.1)).occupied().any()
    checks["something at 1.6 m is above anything this lidar returns: not in the map (a hanging sign there is NOT seen)"] = not grid_with(slab(1.6)).occupied().any()
    m_slab, m_box = grid_with(slab(0.8)), grid_with(box(1.5, 0.0, 0.3, 0.3))
    cx, cy = m_slab.cell(1.97, 0.0)                                                  # 0.32 m from the edge of either
    checks["a person-height obstacle keeps a person's width clear (0.35 m), a dog-height one only the dog's (0.28 m)"] = (
        bool(m_slab.costmap()[1][cy, cx]) and not bool(m_box.costmap()[1][cy, cx]))
    checks["two stray points are noise"] = not grid_with(np.array([[1.0, 0.0, 0.5], [1.0, 0.02, 0.4]])).occupied().any()
    checks["the dog's own body is not an obstacle"] = not grid_with(box(0.0, 0.0, 0.3, 0.5, 200)).occupied().any()
    m = RoomMap(extent=(-4, -4, 4, 4), decay_s=2.0)
    m.update(np.vstack((floor, box(1.5, 0.0, 0.3, 0.6))).astype(np.float32), pose, 0.0)
    m.update(floor.astype(np.float32), pose, 0.5)
    early = m.occupied().any()
    m.update(floor.astype(np.float32), pose, 12.0)
    checks["a removed obstacle fades once it stops being seen (not at once, but within a few seconds)"] = early and not m.occupied().any()
    # -- the real lidar: each message is a persistent map, and nothing standing up is returned nearer than ~1 m
    obst = np.vstack((floor, box(1.5, 0.0, 0.3, 0.6))).astype(np.float32)
    m = RoomMap(extent=(-4, -4, 4, 4), replace=True)
    m.update(obst, pose, 0.0)
    was = m.occupied().any()
    m.update(floor.astype(np.float32), pose, 0.1)
    checks["replace mode: a message that no longer has the obstacle removes it at once (the dog's own map already did the fading)"] = was and not m.occupied().any()
    m = RoomMap(extent=(-4, -4, 4, 4), replace=True, hold_radius=1.1)
    m.update(obst, pose, 0.0)
    m.update(floor.astype(np.float32), (0.8, 0.0, STAND_HEIGHT, 0.0), 0.1)            # the dog walked up to it: the lidar can't see 0.7 m ahead
    held = bool(m.occupied()[m.cell(1.5, 0.0)[1], m.cell(1.5, 0.0)[0]])
    m.update(floor.astype(np.float32), (-1.0, 0.0, STAND_HEIGHT, 0.0), 0.2)           # stepped back out of the blind zone and it is not there
    checks["hold radius: an obstacle seen from afar is remembered while the dog is too close to see it, and dropped once it is not there when it can look"] = (
        held and not m.occupied().any())
    m = RoomMap(extent=(-4, -4, 4, 4), replace=True, hold_radius=1.1)
    m.update(floor.astype(np.float32), pose, 0.0)
    m.update(np.vstack((floor, box(0.7, 0.0, 0.3, 0.6))).astype(np.float32), pose, 0.1)
    checks["hold radius: something that does show up inside it still counts"] = bool(m.occupied()[m.cell(0.7, 0.0)[1], m.cell(0.7, 0.0)[0]])
    m = RoomMap(extent=(-4, -4, 4, 4), decay_s=2.0, hold_radius=1.1)
    m.update(np.vstack((floor, box(0.7, 0.0, 0.3, 0.6))).astype(np.float32), pose, 0.0)
    m.update(floor.astype(np.float32), pose, 20.0)
    checks["hold radius also stops the time decay of a close cell (accumulating mode)"] = bool(m.occupied()[m.cell(0.7, 0.0)[1], m.cell(0.7, 0.0)[0]])
    m = grid_with(box(1.5, 0.5, 0.3, 0.6))
    m.save("_room_test.npz")
    m2 = RoomMap.load("_room_test.npz")
    os.remove("_room_test.npz")
    checks["a saved map loads back identical"] = bool(np.array_equal(m.hits, m2.hits) and np.array_equal(m.seen, m2.seen) and m2.res == m.res and m2.x0 == m.x0)
    m = grid_with(box(1.5, 0.5, 0.3, 0.6), replace=True, hold_radius=1.1)
    m.save("_room_test.npz")
    m2 = RoomMap.load("_room_test.npz")
    os.remove("_room_test.npz")
    checks["...including whether it replaces and its hold radius"] = m2.replace and m2.hold_radius == 1.1
    m = RoomMap(extent=(-4, -4, 4, 4))
    m.update(floor[floor[:, 0] < 0.5].astype(np.float32), pose, 0.0)                    # only the left half was ever looked at
    checks["without allow_unknown a goal in unseen space has no path"] = plan(m, pose, (2.0, 0.0), allow_unknown=False) is None
    checks["with allow_unknown the same goal is reachable"] = plan(m, pose, (2.0, 0.0)) is not None

    # -- drop-offs in the map: a step down 2 m ahead, seen from the start
    def stairs_map(sim, seconds=2.0, **kw):
        m = RoomMap(extent=(-4, -4, 7, 4), decay_s=None, replace=True, hold_radius=1.1, cliffs=True, **kw)
        while sim.t < seconds:
            m.update(sim.scan(), sim.pose(), sim.t)
            sim.t += 0.1
        return m
    sim = PersistentSim([], pits=[(2.0, -4, 7, 4, 0.17)])
    m = stairs_map(sim)
    cx, cy = m.cell(2.1, 0.0)
    checks["a step down 2 m ahead is in the map as a drop-off (with the real lidar's blind zone and floor density)"] = bool(m.cliff[cy, cx]) and not bool(m.cliff[m.cell(1.2, 0.0)[1], m.cell(1.2, 0.0)[0]])
    blocked = m.costmap((0.0, 0.0, STAND_HEIGHT, 0.0))[1]
    bx, by = m.cell(1.5, 0.0)
    checks["...and the planner keeps 0.75 m off it (1.5 m: blocked; 1.0 m: free)"] = bool(blocked[by, bx]) and not bool(blocked[m.cell(1.0, 0.0)[1], m.cell(1.0, 0.0)[0]])
    sim.x = 1.6                                                                       # walk up to it: it is now nearer than the detector looks
    for _ in range(10):
        m.update(sim.scan(), sim.pose(), sim.t)
        sim.t += 0.1
    checks["the dog is now 0.4 m from the edge, closer than it judges, and the map still remembers it"] = bool(m.cliff[cy, cx])
    checks["cliff_ahead sees it 0.5 m in front of the dog"] = (m.cliff_ahead((1.6, 0.0, STAND_HEIGHT, 0.0), 1.0) or 9) < 0.7
    m.save("_room_test.npz")
    m2 = RoomMap.load("_room_test.npz")
    os.remove("_room_test.npz")
    checks["a saved map keeps its drop-offs and overhead layer"] = bool(np.array_equal(m.cliff, m2.cliff) and np.array_equal(m.hits_hi, m2.hits_hi) and m2.cliffs)
    m = stairs_map(PersistentSim([]))
    checks["flat dense floor: no drop-off in the map"] = not m.cliff.any()
    m = stairs_map(Sim([]))
    checks["a sparse floor (46 returns/m^2, single scans, nothing remembered) and no ledge: no drop-off either"] = not m.cliff.any()

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
