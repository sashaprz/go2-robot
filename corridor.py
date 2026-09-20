"""Walk down a corridor: stay in the middle, keep the walls parallel, stop when something is in the way (or the floor ends).

Purely reactive, from ONE lidar scan at a time (no map, no goal): it heads for the most open direction (so slanted
corridors, bends and jogs are just "open to one side"), the walls on its left and right keep it centred when it is
heading straight, and it slows near things. It stops when no direction has room (a dead end, or boxed in), and it
does not plan: it can't choose between two openings or remember where it has been. Bends need a corridor at least ~1 m wide. (That is the map + path planner in
pathplan.py.)

Drop-offs: every scan is also checked for stairs going down or a ledge (dropoff.py: floor that stops in plain sight, or ground well below it), and one is treated as
a wall at its lip, so the walker slows and stops in front of it (state "blocked", .why says "drop-off"). It sees them from about 2.3 m. Things up to ~1.2 m high
count (a table top or shelf across the way stops it); above that this lidar returns nothing, so a low sign or beam is NOT seen.

  python corridor.py                run the self-test (simulated corridors, a dog that lags its commands): no dog needed
  python corridor.py -v             also print each scenario's track
  python corridor.py --live --dry   dog on, Wi-Fi joined: print what it WOULD do, never moves (corridor.bat --dry)
  python corridor.py --live         really walk (corridor.bat). Ctrl-C stops it; so does a stale lidar, or 40 s.

Nothing here has touched the real dog yet. Do the --dry run first, with the dog standing in a corridor.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import dropoff
import obstacles
from obstacles import HEAD_TOP, Corridor, PathWatcher, clearance, to_dog_frame


class CorridorWalker:
    """step(points_in_dog_frame, now) -> (vx, vy, yaw rate). .state is "walking", "blocked" or "no_data".

    Each scan it asks "how far could I walk in each direction (every 10 degrees, +-90)?" and heads for the most open one,
    preferring the direction the walls point (straight, in a straight corridor). So a slanted corridor, a bend or a
    jog sideways just means the most open direction is off to one side, and it turns toward it. It stops only when NO
    direction has `stop_at` m of room (a dead end, or boxed in). Each direction is judged along a strip 0.6 m wide: the
    dog is ~0.31 m wide, so it keeps about 15 cm off the walls."""

    def __init__(self, vmax: float = 0.4, stop_at: float = 0.9, slow_from: float = 1.9, hold: float = 0.6,
                 side_range: float = 1.2, look: float = 2.5, vy_max: float = 0.15, wmax: float = 0.8,
                 k_y: float = 0.6, k_w: float = 1.5, min_points: int = 400, vmin: float = 0.12, prefer_straight: float = 0.8,
                 cliffs: bool = True, cliff_stop: float = 1.3):
        self.vmax, self.stop_at, self.slow_from, self.hold = vmax, stop_at, slow_from, hold
        self.side_range, self.look, self.vy_max, self.wmax, self.k_y, self.k_w = side_range, look, vy_max, wmax, k_y, k_w
        self.vmin = min(vmin, vmax)         # the dog barely moves for a forward command below ~0.1 m/s: walk properly or not at all
        self.prefer_straight = prefer_straight   # m of clearance one radian off the walls' direction is worth
        self.min_points = min_points        # fewer points than this in a whole scan: the lidar is failing, don't walk blind
        self.watch = PathWatcher(stop_at=stop_at, clear_for=1.0)
        self.bearings = np.radians(np.arange(-90, 91, 10))
        self.state, self.left, self.right, self.front, self.tilt, self.steer = "walking", None, None, None, 0.0, 0.0
        self.cliffs = cliffs                                    # look for stairs down / ledges too
        self.cliff_watch = PathWatcher(stop_at=cliff_stop, clear_for=1.0)     # a drop-off in the cone ahead: stops at once whatever the steering thinks
        self.cliff: float | None = None                         # m to the nearest drop-off this scan, None if there is none
        self.why = ""                                           # when "blocked": what is in the way

    def _wall(self, p: np.ndarray, sign: int) -> tuple[float | None, float | None]:
        """(distance to that side's wall in m, its angle to the dog's heading in rad), None where it can't tell."""
        side = sign * p[:, 1]
        q = p[(side > 0.1) & (side < self.side_range) & (p[:, 0] > -0.2) & (p[:, 0] < self.look)]
        if len(q) < 15:
            return None, None
        d = float(np.percentile(sign * q[:, 1], 10))
        face = q[sign * q[:, 1] < d + 0.25]
        if len(face) < 10 or np.ptp(face[:, 0]) < 0.8:                        # not a wall running along the way we walk (an end wall, a chair leg...)
            return None, None
        return d, math.atan(float(np.polyfit(face[:, 0], face[:, 1], 1)[0]))

    def step(self, pts: np.ndarray, now: float) -> tuple[float, float, float]:
        if len(pts) < self.min_points:
            self.state = "no_data"
            return 0.0, 0.0, 0.0
        band = pts[(pts[:, 2] > 0.15) & (pts[:, 2] < HEAD_TOP)]
        band = band[~((np.abs(band[:, 0]) < obstacles_body[0]) & (np.abs(band[:, 1]) < obstacles_body[1]))]   # not the dog itself
        (self.left, la), (self.right, ra) = self._wall(band, +1), self._wall(band, -1)
        angles = [a for a in (la, ra) if a is not None]
        self.tilt = float(np.mean(angles)) if angles else 0.0
        ahead = band                                                            # what the strips look for: obstacles, plus the lip of any drop-off
        self.cliff = None
        if self.cliffs:
            seen, low, obst, origin = dropoff.grids_from_points(pts)
            found = dropoff.detect(seen, low, obst, origin, 0.1, (0.0, 0.0, 0.0))
            self.cliff = found.nearest
            in_cone = None
            if found.nearest is not None:
                xy = dropoff.cells_xy(found, origin, 0.1)
                ahead = np.vstack((band, np.column_stack((xy, np.full(len(xy), 0.5)))))
                ang = np.abs(np.arctan2(xy[:, 1], xy[:, 0]))
                cone = (xy[:, 0] > 0.0) & (ang <= math.radians(60))
                if cone.any():
                    in_cone = float(np.hypot(xy[cone, 0], xy[cone, 1]).min())
            if self.cliff_watch.update(in_cone, now):               # the steering below picks the most OPEN direction, and a lip that spans the corridor
                self.state, self.front = "blocked", in_cone or 0.0   # leaves diagonals toward the side walls looking open: so this is a hard stop
                self.why = "drop-off ahead (stairs down?)"
                return 0.0, 0.0, 0.0
        clear = []
        for th in self.bearings:                                                # free distance along each bearing, for a dog-wide strip
            c = clearance(ahead, Corridor(length=self.look, half_width=0.3, heading=float(th)))
            clear.append(self.look if c is None else c)
        clear = np.asarray(clear)
        if self.watch.update(float(clear.max()), now):                            # nowhere to go: stay put
            self.state, self.front = "blocked", float(clear.max())
            self.why = "drop-off ahead (stairs down?)" if self.cliff is not None and self.cliff < self.stop_at + 1.2 else "something in the way, or a dead end"
            return 0.0, 0.0, 0.0
        self.state, self.why = "walking", ""
        score = clear - self.prefer_straight * np.abs(self.bearings - self.tilt)
        i = int(np.argmax(score))
        th, self.front, self.steer = float(self.bearings[i]), float(clear[i]), float(self.bearings[i])
        wz = 0.0 if abs(th) < 0.05 else max(-self.wmax, min(self.wmax, self.k_w * th))
        vy = 0.0
        if abs(th) < 0.35:                                                        # roughly straight on: also centre between the walls
            if self.left is not None and self.right is not None:
                off = (self.left - self.right) / 2                                # + = nearer the right wall, so slide left
            elif self.left is not None:
                off = self.left - self.hold                                       # only a left wall: keep `hold` m from it
            elif self.right is not None:
                off = -(self.right - self.hold)
            else:
                off = 0.0                                                         # open floor: just go straight
            vy = 0.0 if abs(off) < 0.05 else max(-self.vy_max, min(self.vy_max, self.k_y * off))
        align = math.cos(th)
        if align < 0.35:                                                          # well off to one side: turn on the spot first
            return 0.0, vy, wz
        room = max(0.4, min(1.0, (self.front - self.stop_at) / (self.slow_from - self.stop_at)))
        return max(self.vmin, self.vmax * room * align ** 2), vy, wz


obstacles_body = (0.35, 0.25)        # half-extents of the dog's body: the same box pathplan.BODY uses


# ---- self-test: simulated corridors ------------------------------------------------------------------------
def _polyline_corridor(sim, centre, width):
    """Walls either side of the polyline `centre` (mitred at the corners) plus a cap at each end."""
    P = np.asarray(centre, float)
    d = np.diff(P, axis=0)
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    n = np.stack((-d[:, 1], d[:, 0]), 1)
    m = [n[0]] + [(n[i - 1] + n[i]) / (1 + n[i - 1] @ n[i]) for i in range(1, len(P) - 1)] + [n[-1]]
    for sgn in (+1, -1):
        edge = [tuple(P[i] + sgn * width / 2 * m[i]) for i in range(len(P))]
        for a, b in zip(edge, edge[1:]):
            sim.add_segment(a, b)
    for i in (0, len(P) - 1):
        sim.add_segment(tuple(P[i] + width / 2 * m[i]), tuple(P[i] - width / 2 * m[i]))


def _drive(sim, walker, seconds=40.0, events=(), dt=0.1):
    pending = sorted(events, key=lambda e: e[0])
    states = []
    while sim.t < seconds:
        while pending and sim.t >= pending[0][0]:
            pending.pop(0)[1](sim)
        cmd = walker.step(to_dog_frame(sim.scan(), *sim.pose()), sim.t)
        states.append(walker.state)
        sim.tick(cmd, dt)
    return states


def _selftest(verbose: bool) -> int:
    from pathplan import PersistentSim, Sim

    checks: dict[str, bool] = {}
    end = [(7.0, -4, 7.1, 4, 1.8)]
    both = [(-3, 0.8, 7, 0.9, 1.8), (-3, -0.9, 7, -0.8, 1.8)]           # 1.6 m wide, walls at y = +-0.85, dead end at x = 7

    def show(name, sim):
        if verbose:
            print(f"  [{name}] " + " ".join(f"({x:.1f},{y:.1f})" for x, y in sim.track[::25]))

    sim = Sim(both + end, start=(0.0, 0.5))
    w = CorridorWalker()
    st = _drive(sim, w)
    show("offset start", sim)
    mid = [abs(y) for x, y in sim.track if 2.5 < x < 5.0]
    checks[f"starts 0.5 m off centre: settles on the middle (median offset {np.median(mid):.2f} m)"] = float(np.median(mid)) < 0.15
    checks[f"walks to the dead end and stops there, walls and end wall untouched (x={sim.x:.2f}, closest {sim.gap():.2f} m)"] = (
        5.0 < sim.x < 6.6 and sim.gap() > 0.25 and st[-1] == "blocked")

    sim = Sim(both + end, start=(0.0, 0.0), yaw=math.radians(25))
    w = CorridorWalker()
    _drive(sim, w, seconds=15.0)
    show("yawed start", sim)
    checks[f"starts pointing 25 deg off: straightens up ({math.degrees(sim.yaw):.0f} deg) and doesn't touch a wall ({sim.gap():.2f} m)"] = (
        abs(math.degrees(sim.yaw)) < 6 and sim.gap() > 0.2)

    sim = Sim([(-3, 0.8, 7, 0.9, 1.8)] + end, start=(0.0, 0.0))
    w = CorridorWalker()
    _drive(sim, w, seconds=10.0)                                            # (later it reaches the end wall and turns away from it: right)
    show("one wall", sim)
    d_left = 0.8 - sim.y
    checks[f"one wall only (on the left): holds about 0.6 m from it (now {d_left:.2f} m) and keeps walking (x={sim.x:.1f})"] = (
        0.45 < d_left < 0.8 and sim.x > 3.0)

    block = [(4.0, -0.5, 4.6, 0.5, 0.6)]                                    # leaves 0.3 m each side: too narrow for the dog
    sim = Sim(both + end + block, start=(0.0, 0.0))
    w = CorridorWalker()
    _drive(sim, w)
    show("box in corridor", sim)
    checks[f"a box that blocks the corridor: stops in front of it and stays stopped (x={sim.x:.2f}, state {w.state})"] = (
        w.state == "blocked" and sim.x < 3.6 and sim.gap() > 0.3)

    sim = Sim(both + end + block, start=(0.0, 0.0))
    w = CorridorWalker()
    _drive(sim, w, seconds=40.0, events=[(20.0, lambda s: s.remove_last())])
    checks[f"...and walks on once the box is taken away (x={sim.x:.2f})"] = sim.x > 5.0

    def bendy(name, centre, start, yaw, want, seconds=90.0, vmax=0.2, width=1.4):
        """A corridor along the polyline `centre`; the dog must get within 1.6 m of `want` (the far end) without touching a wall."""
        sim = Sim([], start=start, yaw=yaw)
        _polyline_corridor(sim, centre, width)
        w = CorridorWalker(vmax=vmax)
        _drive(sim, w, seconds=seconds)
        show(name, sim)
        d = math.hypot(sim.x - want[0], sim.y - want[1])
        checks[f"{name}: gets to the far end ({d:.1f} m from it) without touching a wall (closest {sim.gap():.2f} m)"] = d < 1.6 and sim.gap() > 0.2
        return sim, w

    bendy("a corridor that bends 20 deg left", [(-2, 0), (1.5, 0), (1.5 + 5 * math.cos(math.radians(20)), 5 * math.sin(math.radians(20)))],
          (0.0, 0.0), 0.0, (1.5 + 5 * math.cos(math.radians(20)), 5 * math.sin(math.radians(20))))
    bendy("a corridor slanted 20 deg from where the dog faces", [(-2, -0.36 - 0.36), (6, 2.55)], (-1.0, -0.36), 0.0, (6, 2.55))
    bendy("a 90 deg left corner", [(-2, 0), (4.7, 0), (4.7, 6)], (0.0, 0.0), 0.0, (4.7, 6))
    bendy("a 90 deg right corner", [(-2, 0), (4.7, 0), (4.7, -6)], (0.0, 0.0), 0.0, (4.7, -6))
    bendy("a jog sideways (left 30 deg, then straight again)", [(-2, 0), (2, 0), (4, 1.15), (8, 1.15)], (0.0, 0.0), 0.0, (8, 1.15))
    bendy("an S-bend", [(-2, 0), (2, 0), (4, 1.15), (6, 1.15), (8, 0), (11, 0)], (0.0, 0.0), 0.0, (11, 0), seconds=110.0)
    bendy("a narrow 1.0 m corridor with a bend", [(-2, 0), (3, 0), (3 + 3 * math.cos(math.radians(30)), 3 * math.sin(math.radians(30)))],
          (0.0, 0.0), 0.0, (3 + 3 * math.cos(math.radians(30)), 3 * math.sin(math.radians(30))), width=1.0)

    sim = Sim([], start=(0.0, 0.0))
    w = CorridorWalker()
    _drive(sim, w, seconds=6.0)
    checks[f"open floor, no walls: goes straight ahead (y={sim.y:.2f}, yaw {math.degrees(sim.yaw):.1f} deg)"] = abs(sim.y) < 0.1 and abs(sim.yaw) < 0.1 and sim.x > 1.5

    w = CorridorWalker()
    checks["a nearly empty scan (lidar failing) means stop, not walk blind"] = w.step(np.zeros((10, 3), np.float32), 0.0) == (0.0, 0.0, 0.0) and w.state == "no_data"

    sim = Sim([], start=(0.0, 0.0))
    _polyline_corridor(sim, [(-2, 0), (4.7, 0), (4.7, 6)], 1.4)
    w = CorridorWalker(vmax=0.3)
    cmds = []
    while sim.t < 40:
        c = w.step(to_dog_frame(sim.scan(), *sim.pose()), sim.t)
        cmds.append(c)
        sim.tick(c)
    checks["never commands more than vmax forward, 0.15 m/s sideways, 0.8 rad/s turn"] = (
        max(c[0] for c in cmds) <= 0.3 + 1e-9 and max(abs(c[1]) for c in cmds) <= 0.15 + 1e-9 and max(abs(c[2]) for c in cmds) <= 0.8 + 1e-9)
    checks["never asks for a forward speed the dog won't walk at (between 0 and 0.12 m/s)"] = all(c[0] == 0 or c[0] >= 0.12 - 1e-9 for c in cmds)

    # -- stairs down at the end of the corridor, and something up high across it
    stairs = [(4.0, -0.8, 7.0, 0.8, 0.17), (4.6, -0.8, 7.0, 0.8, 0.34)]              # two steps down, no wall at the end: the floor just stops
    sim = PersistentSim(both, pits=stairs, floor_half=7.5, floor_per_m2=200.0)
    w = CorridorWalker()
    _drive(sim, w, seconds=60.0)
    show("stairs down", sim)
    checks[f"the corridor ends in stairs going down: it stops {4.0 - sim.x:.2f} m before the top step, blocked by a '{w.why}'"] = (
        w.state == "blocked" and "drop-off" in w.why and sim.pit_gap() > 0.4)
    sim = PersistentSim(both, pits=stairs, floor_half=7.5, floor_per_m2=200.0)
    w = CorridorWalker(cliffs=False)
    _drive(sim, w, seconds=60.0)
    checks[f"...and with drop-off detection off the same walk goes over the top step (pit gap {sim.pit_gap():.2f} m)"] = sim.pit_gap() < 0

    sim = PersistentSim(both + end, floor_half=7.5, floor_per_m2=200.0)
    w = CorridorWalker()
    _drive(sim, w, seconds=60.0)
    checks[f"an ordinary corridor with a wall at the end on a dense floor: no false drop-off (state {w.state}, why '{w.why}')"] = (
        w.state == "blocked" and "drop-off" not in w.why and sim.x > 4.5)

    sim = PersistentSim(both + end, floor_half=7.5, floor_per_m2=200.0)
    sim.add_slab(3.5, -0.8, 4.1, 0.8, 1.05, 1.15)                                    # a shelf at 1.1 m across the whole corridor: above the old 1.0 m limit
    w = CorridorWalker()
    _drive(sim, w, seconds=60.0)
    checks[f"a shelf at 1.1 m across the corridor: stops before it ({3.5 - sim.x:.2f} m short, the walker's usual standoff), it used to walk under it"] = (
        w.state == "blocked" and sim.x < 3.5)

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


# ---- live ---------------------------------------------------------------------------------------------------
def _live(dry: bool, seconds: float, speed: float) -> int:
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
    if not dry:
        r.sport("BalanceStand")
        time.sleep(1.5)
    walker = CorridorWalker(vmax=speed)
    print(("DRY RUN: nothing will move. " if dry else "WALKING. Ctrl-C stops. ")
          + f"{seconds:.0f} s max. Put the dog in a corridor, facing along it.\n", flush=True)
    t0 = last_print = time.time()
    blocked_since: float | None = None
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.1)
            now = time.time()
            with lock:
                got = latest.get("v")
            if got is None or now - got[0] > 0.5:                    # stale lidar: a dog that can't see must not walk
                if not dry:
                    r.stop_move()
                if now - last_print > 1:
                    last_print = now
                    print("no fresh lidar: standing still", flush=True)
                if now - (got[0] if got else t_lidar) > 4 and now - last_kick > 6:      # the dog's lidar sometimes goes quiet: reset it (off, then on)
                    last_kick = now
                    print("lidar silent: switching it off and on again", flush=True)
                    pub, topic = r.c.conn.datachannel.pub_sub.publish_without_callback, r._topic["ULIDAR_SWITCH"]
                    r.c.loop.call_soon_threadsafe(pub, topic, "off")
                    time.sleep(0.8)
                    r.c.loop.call_soon_threadsafe(pub, topic, "on")
                continue
            cmd = walker.step(to_dog_frame(got[1], *got[2]), now)
            if not dry:
                r.move(*cmd) if any(cmd) else r.stop_move()
            if now - last_print > 1:
                last_print = now
                f = lambda v: "  -  " if v is None else f"{v:4.2f} m"      # noqa: E731
                print(f"{walker.state:8s} left {f(walker.left)} right {f(walker.right)} open {f(walker.front)} drop-off {f(walker.cliff)} "
                      f"steer {math.degrees(walker.steer):+4.0f} deg  -> vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} yaw {cmd[2]:+.2f}  {walker.why}", flush=True)
            if walker.state == "blocked":
                blocked_since = blocked_since or now
                if now - blocked_since > 3.0:
                    print(f"stopped: {walker.why}")
                    break
            else:
                blocked_since = None
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        try:
            r.stop_move()
        except Exception:  # noqa: BLE001
            pass
        r.close()
    return 0


if __name__ == "__main__":
    if "--live" in sys.argv:
        sp = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 0.3
        raise SystemExit(_live("--dry" in sys.argv, 40.0, sp))
    raise SystemExit(_selftest("-v" in sys.argv))
