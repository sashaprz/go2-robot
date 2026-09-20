"""Guide mode: lead the dog from where it stands (A) to a point B given as coordinates, round whatever its lidar sees.

B is metres from where the dog is standing when it starts: --goto 4,1 means 4 m ahead and 1 m to the left. Several points
(--goto 4,0 4,3) are visited in order. The dog builds a top-down map from its lidar (pathplan.RoomMap), plans the cheapest path
that stays a dog's width clear of obstacles and prefers the middle of a corridor (A*), follows it, and re-plans twice a second, so
a box that turns up mid-walk is walked round. When there is no way through it stands still and waits for one; it gives up after
`patience` seconds and says why.

What the real dog's lidar turned out to be (recordings, README "Corridor walking"), and what is done about it:
  - each message is the dog's own persistent map, not one scan       -> the grid is rebuilt from each message (RoomMap replace=True)
  - nothing standing up is returned nearer than ~1 m                 -> cells within `blind` of the dog keep what was seen from further off
  - a thing that left lingers in the map for ~40 s                   -> it just looks like an obstacle: the dog waits (patience 90 s)
  - nothing above ~1.2 m comes back                                  -> a hanging sign, a low beam or a low ceiling above that is NOT SEEN (not "clear").
  - the ground past a ledge is hidden by its lip                     -> dropoff.py: floor that stops in plain sight, or ground well below it,
                                                                       is a drop-off: no-go with a 0.75 m margin, and it will not walk toward one
  - a table top or shelf the dog fits under would hit the person     -> anything up to ~1.2 m counts, with a person-wide (0.35 m) margin

  python guide.py                       self-test: simulated rooms and a simulated dog with the real lidar's quirks (no dog needed)
  python guide.py -v                    also draw each scenario
  python guide.py --live --dry --goto 4,1   dog on, Wi-Fi joined: plans and prints what it WOULD do; never moves
  python guide.py --live --goto 4,1         really walks (guide.bat). Ctrl-C stops it; so does a stale lidar, a dog that isn't moving, or the time limit.
  --no-recover                              never back away to look again (someone may be standing right behind the dog)

Nothing here has touched the real dog yet. Do the --dry run first and check the map picture against the room.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque

import numpy as np

from pathplan import Navigator, PersistentSim, RoomMap, Sim, ascii_map


def to_world(start_pose, local: tuple[float, float]) -> tuple[float, float]:
    """(metres ahead, metres to the left) of the pose -> the same point in the frame the pose is in."""
    x, y, _, yaw = start_pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + local[0] * c - local[1] * s, y + local[0] * s + local[1] * c


def to_local(start_pose, world: tuple[float, float]) -> tuple[float, float]:
    x, y, _, yaw = start_pose
    dx, dy = world[0] - x, world[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c + dy * s, -dx * s + dy * c


class Guide:
    """feed(points_world, pose, t) with every lidar message, step(pose, now) -> (vx, vy, yaw rate) at ~10 Hz.

    .state: "going", "waiting" (no way through right now), "recovering" (backing away to look again), "no_data" (lidar stale or
    nearly empty: stands still), then the finished states "arrived", "gave_up" (no path for `patience` s), "stuck" (told to walk,
    not moving). .reason says why.

    Drop-offs (dropoff.py) go in the map as no-go, and on top of that it will not walk toward one that is within `cliff_stop` m of
    its centre even if the planner says go. Recovery: an obstacle that is inside the lidar's ~1 m blind zone is only REMEMBERED, and the
    map does not believe it has gone until the dog can look again. So after `recover_after` s of waiting, if something in its way (or a drop-off) is that close,
    it backs straight away, slowly, just far enough to see past the blind zone, and looks again. Once per wait. It only backs into space the
    map knows is empty; it cannot see a person standing right behind it (that is inside the blind zone too), so pass recover=False if
    someone will be there."""

    def __init__(self, start_pose, goals_local, vmax: float = 0.4, vmin: float = 0.12, wmax: float = 0.8, goal_tol: float = 0.3,
                 patience: float = 90.0, stuck_s: float = 6.0, min_points: int = 400, stale_s: float = 0.6,
                 persistent: bool = True, blind: float = 1.1, margin: float = 4.0, check_stuck: bool = True,
                 cliffs: bool = True, cliff_stop: float = 0.6, recover: bool = True, recover_after: float = 5.0,
                 back_speed: float = 0.15, max_back: float = 1.2):
        self.start = tuple(start_pose)
        self.goals = [to_world(self.start, g) for g in goals_local]
        self.vmax, self.vmin, self.wmax, self.goal_tol = vmax, min(vmin, vmax), wmax, goal_tol
        self.patience, self.stuck_s, self.min_points, self.stale_s = patience, stuck_s, min_points, stale_s
        self.check_stuck = check_stuck                       # off for a dry run: the dog isn't sent anything, so it can't be "stuck"
        self.cliff_stop, self.recover, self.recover_after, self.back_speed, self.max_back = cliff_stop, recover, recover_after, back_speed, max_back
        xs, ys = [self.start[0]] + [g[0] for g in self.goals], [self.start[1]] + [g[1] for g in self.goals]
        extent = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        # the real dog: every message is its persistent map, and it cannot see nearer than ~1 m. A test lidar that sends single scans: decay instead.
        self.room = RoomMap(extent=extent, replace=persistent, hold_radius=blind if persistent else 0.0, decay_s=None if persistent else 4.0,
                            cliffs=cliffs)
        self.leg = 0
        self.nav = Navigator(self.room, self.goals[0], goal_tol=goal_tol, vmax=vmax, wmax=wmax)
        self.state, self.reason = "going", ""
        self.n_points, self.last_feed = 0, -math.inf
        self._waiting_since: float | None = None
        self._backed = False                               # already backed away once during this wait
        self._rec: tuple | None = None                     # (t, x, y, how far) while backing away
        self._hist: deque = deque()                        # (t, x, y, was told to walk) for the stuck check

    @property
    def path(self):
        return self.nav.path

    @property
    def goal(self):
        return self.goals[min(self.leg, len(self.goals) - 1)]

    @property
    def finished(self) -> bool:
        return self.state in ("arrived", "gave_up", "stuck")

    def feed(self, points_world: np.ndarray, pose, t: float) -> None:
        self.n_points, self.last_feed = len(points_world), t
        if self.n_points >= self.min_points:                # a nearly empty message means the lidar is failing: don't learn "everything is free" from it
            self.room.update(points_world, pose, t)

    def _back_distance(self, pose) -> float:
        """How far to back away so what is in the way is no longer inside the blind zone (or too near a drop-off): 0 if nothing is."""
        need = 0.0
        hold = self.room.hold_radius
        if hold > 0:
            d = self.room.nearest_ahead(self.room.occupied() | self.room.occupied_high(), pose, hold + 0.15, half_width=0.6)
            if d is not None:
                need = hold + 0.15 - d
        d = self.room.nearest_ahead(self.room.cliff, pose, 0.95, half_width=0.6) if self.room.cliff.any() else None
        if d is not None:
            need = max(need, 0.95 - d)                       # the planner keeps 0.75 m from a drop-off: back to 0.2 m outside that
        return min(need, self.max_back)

    def _back_up(self, pose, now: float) -> tuple[float, float, float]:
        t0, x0, y0, need = self._rec
        moved = math.hypot(pose[0] - x0, pose[1] - y0)
        if moved >= need - 0.02 or now - t0 > need / (0.6 * self.back_speed) + 3.0:
            self._rec = None
            self.state, self.reason = "waiting", "backed away: looking again"
            return 0.0, 0.0, 0.0
        self.state, self.reason = "recovering", f"backing away {moved:.2f} of {need:.2f} m to look again"
        return -self.back_speed, 0.0, 0.0

    def step(self, pose, now: float) -> tuple[float, float, float]:
        if self.finished:
            return 0.0, 0.0, 0.0
        if now - self.last_feed > self.stale_s or self.n_points < self.min_points:
            if self.state != "no_data":
                self.reason = "lidar stale" if now - self.last_feed > self.stale_s else f"only {self.n_points} lidar points"
            self.state, self._rec = "no_data", None
            self._hist.clear()
            return 0.0, 0.0, 0.0
        if self._rec is not None:
            return self._back_up(pose, now)
        cmd = self.nav.step(pose, now)
        if self.nav.state == "arrived":
            self.leg += 1
            if self.leg >= len(self.goals):
                self.state, self.reason = "arrived", f"{math.hypot(self.goals[-1][0] - pose[0], self.goals[-1][1] - pose[1]):.2f} m from the goal"
                return 0.0, 0.0, 0.0
            self.nav = Navigator(self.room, self.goals[self.leg], goal_tol=self.goal_tol, vmax=self.vmax, wmax=self.wmax)
            self._hist.clear()
            cmd = self.nav.step(pose, now)
        blocked, why = self.nav.state == "no_path", ""
        if not blocked and cmd[0] > 0.0:                     # the planner says go: a drop-off right in front of the nose still overrides it
            d = self.room.cliff_ahead(pose, self.cliff_stop + cmd[0])
            if d is not None:
                blocked, why = True, f"drop-off {d:.1f} m ahead"
        if blocked:
            if self._waiting_since is None:
                self._waiting_since, self._backed = now, False
            waited = now - self._waiting_since
            hazard = self.room.nearest_any(self.room.cliff, pose, 2.0) if self.room.cliff.any() else None      # a drop-off within 2 m, whichever way the dog faces
            note = why or (f"no way through for {waited:.0f} s" + (f", drop-off {hazard:.1f} m ahead" if hazard is not None else ""))
            self.state, self.reason = "waiting", note
            self._hist.clear()
            if waited >= self.patience:
                self.state = "gave_up"
                self.reason = f"no way to the goal for {self.patience:.0f} s (" + ("a drop-off is in the way" if hazard is not None else "blocked, or boxed in") + ")"
            elif self.recover and not self._backed and waited >= self.recover_after:
                need = self._back_distance(pose)
                if need >= 0.05:
                    self._backed = True                      # once per wait, whether or not there is room to do it
                    if self.room.free_behind(pose, need + 0.3):
                        self._rec = (now, pose[0], pose[1], need)
                        return self._back_up(pose, now)
                    self.reason = note + " (can't back away: something behind)"
            return 0.0, 0.0, 0.0
        self._waiting_since, self._backed = None, False
        self.state, self.reason = "going", ""
        vx, vy, wz = cmd
        if 0.0 < vx < self.vmin:                             # the dog barely moves under ~0.1 m/s: walk properly, or turn on the spot
            vx = 0.0
        self._hist.append((now, pose[0], pose[1], vx >= self.vmin))
        while self._hist and now - self._hist[0][0] > self.stuck_s:
            self._hist.popleft()
        if self.check_stuck and len(self._hist) > 5 and now - self._hist[0][0] > 0.9 * self.stuck_s:     # told to walk for most of the last few seconds, and hasn't gone anywhere
            walked = sum(b[0] - a[0] for a, b in zip(self._hist, list(self._hist)[1:]) if a[3])
            moved = math.hypot(pose[0] - self._hist[0][1], pose[1] - self._hist[0][2])
            if walked > 0.6 * self.stuck_s and moved < 0.15:
                self.state, self.reason = "stuck", f"told to walk for {walked:.0f} s and moved {moved * 100:.0f} cm (something the lidar can't see?)"
                return 0.0, 0.0, 0.0
        return vx, vy, wz


# ---- self-test: simulated rooms ------------------------------------------------------------------------------
def _drive(sim, guide: Guide, seconds: float = 60.0, events=(), dt: float = 0.1, lidar_out=lambda t: False, frozen: bool = False):
    """Run guide against sim. Returns the list of (state, cmd) per tick. lidar_out(t) True = no message that tick; frozen = the dog never moves."""
    pending = sorted(events, key=lambda e: e[0])
    log = []
    while sim.t < seconds:
        while pending and sim.t >= pending[0][0]:
            pending.pop(0)[1](sim)
        if not lidar_out(sim.t):
            guide.feed(sim.scan(), sim.pose(), sim.t)
        cmd = guide.step(sim.pose(), sim.t)
        log.append((guide.state, cmd))
        if guide.finished:
            break
        if frozen:
            sim.t += dt
        else:
            sim.tick(cmd, dt)
    return log


def _selftest(verbose: bool) -> int:
    checks: dict[str, bool] = {}
    walls = [(-6, -4, 6, -3.9, 1.8), (-6, 3.9, 6, 4, 1.8), (-6, -4, -5.9, 4, 1.8), (5.9, -4, 6, 4, 1.8)]

    def show(name, sim, guide):
        if verbose:
            print(f"  [{name}]")
            print(ascii_map(guide.room, sim.pose(), guide.path, guide.goal))
            print(f"    t={sim.t:.1f}s state={guide.state} ({guide.reason}) at ({sim.x:.2f}, {sim.y:.2f}) closest approach {sim.gap():.2f} m")

    # -- coordinates: A is wherever the dog stands and faces, B is metres ahead / to its left
    p0 = (10.0, -5.0, 0.32, math.pi / 2)
    w = to_world(p0, (2.0, 1.0))
    checks[f"a start pose facing +y at (10, -5): 2 m ahead and 1 m left is world (9, -3) (got {w[0]:.2f}, {w[1]:.2f})"] = abs(w[0] - 9) < 1e-9 and abs(w[1] + 3) < 1e-9
    back = to_local(p0, w)
    checks["...and to_local undoes it"] = abs(back[0] - 2) < 1e-9 and abs(back[1] - 1) < 1e-9

    false_alarms: list[bool] = []

    def run(name, sim, goals, seconds=60.0, events=(), lidar_out=lambda t: False, frozen=False, **kw):
        g = Guide(sim.pose(), goals, persistent=isinstance(sim, PersistentSim), **kw)
        log = _drive(sim, g, seconds, events, lidar_out=lidar_out, frozen=frozen)
        show(name, sim, g)
        if not sim.pits:
            false_alarms.append(bool(g.room.cliff.any()))          # a floor with no ledge must never end up with a drop-off in the map
        return g, log

    sim = PersistentSim(walls, start=(2.0, 1.0), yaw=math.pi / 2)
    g, _ = run("start pose", sim, [(2.0, 1.0)])                                      # 2 m ahead (world +y), 1 m to its left (world -x)
    checks[f"the dog starts at (2, 1) facing +y: '2 m ahead, 1 m left' ends at world (1, 3) (got {sim.x:.2f}, {sim.y:.2f})"] = (
        g.state == "arrived" and math.hypot(sim.x - 1.0, sim.y - 3.0) < 0.4)

    sim = PersistentSim(walls)
    g, log = run("open floor", sim, [(4.0, 0.0)])
    checks[f"open floor, B is 4 m ahead: arrives ({sim.t:.1f} s), in a straight-ish line"] = g.state == "arrived" and sim.t < 25 and max(abs(y) for _, y in sim.track) < 0.5
    checks["never commands more than the top speed, or a forward speed the dog won't walk at (0 < vx < 0.12)"] = (
        max(c[0] for _, c in log) <= 0.4 + 1e-9 and all(c[0] == 0 or c[0] >= 0.12 - 1e-9 for _, c in log) and max(abs(c[2]) for _, c in log) <= 0.8 + 1e-9)

    sim = PersistentSim(walls + [(1.6, -0.3, 2.2, 0.3, 0.6)])
    g, _ = run("a box in the way", sim, [(4.0, 0.0)])
    checks[f"a box in the way (with the real lidar: blind under 1 m): goes round it and arrives, {sim.gap():.2f} m clear at the closest"] = g.state == "arrived" and sim.gap() > 0.18

    # the pessimistic lidar: the dog's map also drops what is inside the blind zone, so anything the dog is walking past vanishes
    box = [(1.6, -0.3, 2.2, 0.3, 0.6)]
    sim = PersistentSim(walls + box, forgets_blind=True)
    g, _ = run("map forgets the blind zone, hold radius on", sim, [(4.0, 0.0)])
    gap_hold = sim.gap()
    checks[f"if the dog's map forgets what is nearer than 1 m: the hold radius keeps the box in mind while it walks past ({gap_hold:.2f} m clear)"] = (
        g.state == "arrived" and gap_hold > 0.18)
    sim = PersistentSim(walls + box, forgets_blind=True)
    g0 = Guide(sim.pose(), [(4.0, 0.0)], persistent=True, blind=0.0)
    _drive(sim, g0)
    checks[f"...and without it the dog cuts inside its own 0.28 m safety margin ({sim.gap():.2f} m from the box, against {gap_hold:.2f} m with it)"] = sim.gap() < 0.28 <= gap_hold

    sim = Sim(walls + [(1.6, -0.3, 2.2, 0.3, 0.6)])
    g, _ = run("plain scans", sim, [(4.0, 0.0)])
    checks[f"a lidar that sends single scans instead (decay mode) works too: arrives, {sim.gap():.2f} m clear"] = g.state == "arrived" and sim.gap() > 0.18

    door = walls + [(2.95, -4, 3.05, 0.5, 1.8), (2.95, 1.5, 3.05, 4, 1.8)]
    sim = PersistentSim(door)
    g, _ = run("doorway", sim, [(5.0, -1.0)], seconds=90.0)
    cross = [y for (x, y), (x2, _) in zip(sim.track, sim.track[1:]) if x < 3.0 <= x2]
    checks[f"B is through a 1 m doorway (across the room): finds it (crossed at y={cross[0]:.2f}) and arrives" if cross else "doorway: never crossed the wall"] = (
        g.state == "arrived" and bool(cross) and 0.5 < cross[0] < 1.5 and sim.gap() > 0.18)

    slit = walls + [(2.95, -4, 3.05, 0.6, 1.8), (2.95, 1.0, 3.05, 4, 1.8)]
    sim = PersistentSim(slit)
    g, _ = run("too narrow", sim, [(5.0, 0.0)], seconds=60.0, patience=10.0)
    checks[f"a gap too narrow for the dog: it stands and waits, then gives up ({g.state}: {g.reason}) on its own side (x={sim.x:.2f})"] = g.state == "gave_up" and sim.x < 2.9

    late = (6.0, lambda s: s.add((3.6, -0.4, 4.2, 0.4, 0.8)))
    sim = PersistentSim(walls)
    g, _ = run("box appears", sim, [(5.0, 0.0)], events=[late])
    checks[f"a box appears in the way while it walks: re-plans round it and arrives, {sim.gap():.2f} m clear"] = g.state == "arrived" and sim.gap() > 0.15

    hall = walls + [(1, 0.8, 7, 0.9, 1.8), (1, -0.9, 7, -0.8, 1.8)]                     # 1.6 m wide corridor, x from 1 to 7
    block = (3.5, -0.4, 4.1, 0.4, 0.9)
    sim = PersistentSim(hall + [block])
    g, log = run("corridor blocked, then cleared", sim, [(5.5, 0.0)], seconds=150.0, events=[(25.0, lambda s: s.remove_last())])
    waited = sum(1 for st, _ in log if st == "waiting")
    checks[f"a box fills the corridor: it waits ({waited / 10:.0f} s), never touches it, and walks on once it's gone (the dog's map keeps a ghost for 40 s): {g.state}"] = (
        g.state == "arrived" and waited > 100 and sim.gap() > 0.18)

    sim = PersistentSim(walls)
    g, log = run("lidar drops out", sim, [(5.0, 0.0)], lidar_out=lambda t: 5.0 <= t < 7.0)
    dead = [c for st, c in log if st == "no_data"]
    checks[f"the lidar goes quiet for 2 s: it stands still ({len(dead)} ticks of no_data, all zero) and carries on after ({g.state})"] = (
        len(dead) > 10 and all(c == (0.0, 0.0, 0.0) for c in dead) and g.state == "arrived")

    sim = PersistentSim(walls)
    g = Guide(sim.pose(), [(5.0, 0.0)], persistent=True)
    checks["a nearly empty message (lidar failing) means stop, not walk blind"] = (
        g.feed(np.zeros((10, 3), np.float32), sim.pose(), 0.0) is None and g.step(sim.pose(), 0.0) == (0.0, 0.0, 0.0) and g.state == "no_data")

    sim = PersistentSim(walls)
    g, _ = run("stuck", sim, [(5.0, 0.0)], seconds=30.0, frozen=True)
    checks[f"told to walk but not moving (something the lidar can't see holds it): stops as {g.state} after {sim.t:.0f} s: {g.reason}"] = g.state == "stuck" and sim.t < 12

    sim = PersistentSim(walls)
    g, _ = run("two legs", sim, [(3.0, 0.0), (3.0, 2.5)], seconds=90.0)
    checks[f"two points in a row (3 m ahead, then 2.5 m to the left of that): does both (at {sim.x:.2f}, {sim.y:.2f})"] = (
        g.state == "arrived" and math.hypot(sim.x - 3.0, sim.y - 2.5) < 0.4)

    # ==== drop-offs: stairs down, a hole in the floor. The lidar cannot see past a lip; the map learns a "no-go" from the shadow ====
    stairs = [(2.5, -4.0, 6.0, 4.0, 0.17), (3.2, -4.0, 6.0, 4.0, 0.34)]           # two steps down across the whole room, from x = 2.5
    sim = PersistentSim(walls, pits=stairs)
    g, _ = run("stairs down", sim, [(5.0, 0.0)], seconds=90.0, patience=20.0)
    before = 2.5 - sim.x
    checks[f"stairs going down across the room, B beyond them: stops {before:.2f} m before the top step, never goes over ({g.state}: {g.reason})"] = (
        g.state == "gave_up" and 0.5 < before < 2.0 and sim.pit_gap() > 0.4 and "drop-off" in g.reason)
    sim = PersistentSim(walls, pits=stairs)
    g, _ = run("stairs down, detector off", sim, [(5.0, 0.0)], seconds=60.0, cliffs=False)
    checks[f"...and with drop-off detection switched off the same walk goes straight down them (pit gap {sim.pit_gap():.2f} m)"] = sim.pit_gap() < 0

    hole = [(2.0, -1.0, 3.5, 1.0, 0.3)]
    sim = PersistentSim(walls, pits=hole)
    g, _ = run("hole in the floor", sim, [(5.0, 0.0)], seconds=90.0)
    checks[f"a 30 cm hole in the middle of the room, B beyond: goes round it and arrives, {sim.pit_gap():.2f} m from the edge at the closest"] = (
        g.state == "arrived" and sim.pit_gap() > 0.4)

    sim = PersistentSim(walls, pits=[(2.3, -4.0, 6.0, 4.0, 0.17)], start=(1.9, 0.0))
    g, _ = run("starts at the edge", sim, [(5.0, 0.0)], seconds=90.0, patience=25.0)
    checks[f"starting 0.4 m from a ledge: it does not step toward it, and backs off to {2.3 - sim.x:.2f} m from it ({g.state})"] = (
        sim.pit_gap() > 0.3 and 2.3 - sim.x > 0.75)

    sim = PersistentSim(walls)
    g = Guide(sim.pose(), [(5.0, 0.0)], persistent=True)
    g.feed(sim.scan(), sim.pose(), 0.0)
    g.room.cliff[g.room.cell(0.5, 0.0)[1], g.room.cell(0.5, 0.0)[0]] = True             # a drop-off 0.5 m in front of the nose...
    g.nav.step = lambda pose, now: (0.3, 0.0, 0.0)                                       # ...and a planner that (wrongly) says "go"
    g.nav.state = "going"
    cmd = g.step(sim.pose(), 0.1)
    checks[f"even if the planner says go, a drop-off 0.5 m ahead stops the dog ({g.state}: {g.reason})"] = cmd == (0.0, 0.0, 0.0) and "drop-off" in g.reason

    # ==== up high: a table top the dog fits under, a shelf above the old 1 m limit, a sign above everything the lidar returns ====
    sim = PersistentSim(walls)
    sim.add_slab(2.0, -0.6, 3.2, 0.6, 0.70, 0.80)
    g, _ = run("table", sim, [(5.0, 0.0)], seconds=90.0)
    checks[f"a table top at 0.75 m dead ahead (the dog could walk under it, the person behind it couldn't): goes round, {sim.gap():.2f} m clear"] = (
        g.state == "arrived" and sim.gap() > 0.3)
    sim = PersistentSim(walls)
    sim.add_slab(2.0, -0.6, 3.2, 0.6, 1.05, 1.15)
    g, _ = run("shelf", sim, [(5.0, 0.0)], seconds=90.0)
    checks[f"a shelf at 1.1 m (the old code ignored everything above 1.0 m): goes round, {sim.gap():.2f} m clear"] = g.state == "arrived" and sim.gap() > 0.3
    sim = PersistentSim(walls)
    sim.add_slab(2.0, -0.6, 3.2, 0.6, 1.45, 1.6)
    g, _ = run("sign", sim, [(5.0, 0.0)], seconds=90.0)
    under = min(math.hypot(max(2.0 - x, 0, x - 3.2), max(-0.6 - y, 0, y - 0.6)) for x, y in sim.track)
    checks[f"THE LIMIT: a sign at 1.5 m is above everything this lidar returns, so it walks straight under it ({under:.2f} m of it): not seen, not avoided"] = (
        g.state == "arrived" and under < 0.2)

    # ==== back away and look again ====
    hall = walls + [(1, 0.8, 7, 0.9, 1.8), (1, -0.9, 7, -0.8, 1.8)]
    person = (1.05, -0.4, 1.45, 0.4, 1.6)                                               # steps into the corridor 1.05 m ahead: just inside the blind zone's edge
    leaves = [(0.5, lambda s: s.add(person)), (15.0, lambda s: s.remove_last())]
    sim = PersistentSim(hall)
    g, log = run("back away", sim, [(5.5, 0.0)], seconds=120.0, patience=80.0, events=leaves)
    backed = sum(1 for st, _ in log if st == "recovering")
    checks[f"someone steps in 1.05 m ahead and leaves: it backs away ({backed / 10:.0f} s), looks again, and walks on ({g.state} at t={sim.t:.0f} s)"] = (
        backed > 0 and g.state == "arrived" and sim.gap() > 0.18)
    sim = PersistentSim(hall)
    g, log = run("no recovery", sim, [(5.5, 0.0)], seconds=120.0, patience=80.0, events=leaves, recover=False)
    checks[f"...without the recovery the same thing leaves it waiting on a person who is long gone, until it gives up ({g.state}, x={sim.x:.1f})"] = (
        g.state == "gave_up" and sim.x < 1.0)
    sim = PersistentSim(hall)
    g, log = run("back away, speeds", sim, [(5.5, 0.0)], seconds=60.0, patience=80.0, events=leaves)
    back = [c for st, c in log if st == "recovering"]
    checks[f"it backs at {back[0][0]:+.2f} m/s, straight, and never further than 1.2 m" if back else "it never backed"] = (
        bool(back) and all(c == (-0.15, 0.0, 0.0) for c in back) and len(back) * 0.1 * 0.15 < 1.3)

    checks[f"none of the {len(false_alarms)} runs on a floor with no ledge (walls, boxes, doorways, a lidar dropout...) ever put a drop-off in the map"] = not any(false_alarms)

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


# ---- live ---------------------------------------------------------------------------------------------------
def _live(dry: bool, goals, odom: bool, seconds: float, speed: float, patience: float, recover: bool = True) -> int:
    import os
    import threading

    key = os.environ.get("UNITREE_AES_128_KEY")
    if not key:
        print("UNITREE_AES_128_KEY isn't set (expected in ~/.dimos.env)")
        return 2
    from go2 import Robot                                           # the same connection wrapper the main app uses

    ip = os.environ.get("ROBOT_IP", "192.168.12.1")
    print(f"connecting to {ip} ...", flush=True)
    r = Robot(ip, key)
    latest: dict = {}
    lock = threading.Lock()

    def on_lidar(pts, pose):
        with lock:
            latest["v"] = (time.time(), pts, pose)

    r.on_lidar(on_lidar)
    t_lidar = last_kick = time.time()
    try:
        while "v" not in latest:                                     # need one message to know where "here" is
            if time.time() - t_lidar > 20:
                print("no lidar in 20 s: this dog isn't sending it (or it needs resetting). Nothing to plan from.")
                return 2
            time.sleep(0.2)
        start = latest["v"][2]
        local = [(g[0], g[1]) for g in goals] if not odom else [to_local(start, (g[0], g[1])) for g in goals]
        guide = Guide(start, local, vmax=speed, patience=patience, check_stuck=not dry, recover=recover)
        print(f"start (dog's frame): x {start[0]:.2f} y {start[1]:.2f} heading {math.degrees(start[3]):.0f} deg", flush=True)
        for i, (g, w) in enumerate(zip(local, guide.goals), 1):
            print(f"goal {i}: {g[0]:.2f} m ahead, {g[1]:+.2f} m left of the start  (dog's frame {w[0]:.2f}, {w[1]:.2f})", flush=True)
        if not dry:
            r.sport("BalanceStand")
            time.sleep(1.5)
        print(("DRY RUN: nothing will move. " if dry else "WALKING. Ctrl-C stops. ") + f"{seconds:.0f} s max, top speed {speed} m/s.\n", flush=True)
        t0 = last_print = last_map = last_msg = time.time()
        seen_msg = None
        cmd = (0.0, 0.0, 0.0)
        pose = start
        while time.time() - t0 < seconds:
            time.sleep(0.1)
            now = time.time()
            with lock:
                got = latest.get("v")
            if got is not None and got[0] != seen_msg:                # a new lidar message: learn from it
                seen_msg, last_msg = got[0], got[0]
                pose = got[2]
                guide.feed(got[1], pose, now)
            if now - last_msg > 0.5 and not dry:                       # a dog that can't see must not walk
                r.stop_move()
            if now - last_msg > 4 and now - last_kick > 6:             # the dog's lidar sometimes goes quiet: reset it (off, then on)
                last_kick = now
                print("lidar silent: switching it off and on again", flush=True)
                pub, topic = r.c.conn.datachannel.pub_sub.publish_without_callback, r._topic["ULIDAR_SWITCH"]
                r.c.loop.call_soon_threadsafe(pub, topic, "off")
                time.sleep(0.8)
                r.c.loop.call_soon_threadsafe(pub, topic, "on")
            cmd = guide.step(pose, now)
            if not dry:
                r.move(*cmd) if any(cmd) else r.stop_move()
            if now - last_print > 1:
                last_print = now
                f, l = to_local(start, pose[:2])
                left = math.hypot(guide.goal[0] - pose[0], guide.goal[1] - pose[1])
                print(f"{guide.state:8s} at {f:+5.2f} ahead {l:+5.2f} left of start, {left:4.2f} m to goal {min(guide.leg, len(local) - 1) + 1}/{len(local)}, "
                      f"path {len(guide.path or [])} pts, lidar {guide.n_points} pts, drop-off cells {int(guide.room.cliff.sum())}  -> vx {cmd[0]:+.2f} yaw {cmd[2]:+.2f}  {guide.reason}", flush=True)
            if dry and now - last_map > 5:
                last_map = now
                print("\n" + ascii_map(guide.room, pose, guide.path, guide.goal) + "\n   # obstacle  + too close to pass  v drop-off  . floor  * planned path  D dog  G goal\n", flush=True)
            if guide.finished:
                print(f"\n{guide.state}: {guide.reason}", flush=True)
                break
        else:
            print("\nout of time")
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        try:
            r.stop_move()
        except Exception:  # noqa: BLE001
            pass
        r.close()
    return 0


def _goal(s: str) -> tuple[float, float]:
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' is not AHEAD,LEFT in metres (for example 4,1 or 3,-2)") from None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="connect to the dog (default: run the self-test)")
    ap.add_argument("--dry", action="store_true", help="with --live: plan and print, never move")
    ap.add_argument("--goto", nargs="+", type=_goal, metavar="AHEAD,LEFT", default=[(3.0, 0.0)],
                    help="where to lead the dog: metres ahead and to the left of where it stands now (default 3,0). Several = in order.")
    ap.add_argument("--odom", action="store_true", help="read --goto as x,y in the dog's own odometry frame instead (it resets when the dog reboots)")
    ap.add_argument("--speed", type=float, default=0.3, help="top forward speed, m/s (default 0.3)")
    ap.add_argument("--seconds", type=float, default=120.0, help="give up after this long (default 120)")
    ap.add_argument("--patience", type=float, default=90.0, help="seconds to wait with no way through before giving up (default 90: the dog's map keeps a ghost ~40 s after something leaves)")
    ap.add_argument("--no-recover", action="store_true", help="never back away to look again (use it if someone may be standing right behind the dog)")
    ap.add_argument("-v", action="store_true", help="self-test: draw each scenario")
    a = ap.parse_args()
    if a.live:
        raise SystemExit(_live(a.dry, a.goto, a.odom, a.seconds, a.speed, a.patience, not a.no_recover))
    raise SystemExit(_selftest(a.v))
