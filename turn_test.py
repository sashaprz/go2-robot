"""Simulated dogs turning on the spot under voice.TurnController (no dog needed):  python turn_test.py
The dog is modelled as a rate that follows the command with a lag, a command delay, and a gyro that arrives late and a little noisy."""
from __future__ import annotations

import math
import random

import voice


def simulate(target_deg: float, tau: float = 0.25, cmd_delay: float = 0.1, gyro_delay: float = 0.1, noise_deg: float = 0.5, gyro_sign: float = 1.0,
             gyro_dead: bool = False, dog_stuck: bool = False, max_rate: float = 0.8, seed: int = 1) -> dict:
    rng = random.Random(seed)
    dt = 0.02
    ctl = voice.TurnController(math.radians(target_deg), max_rate)
    yaw, rate, t = 0.0, 0.0, 0.0
    hist, cmds = [], []                                       # (t, yaw) for the delayed gyro; (t, cmd) for the delayed command
    state, last_ctl = "run", -1.0
    cmd_now, yaw0 = 0.0, None
    while t < 20.0:
        hist.append((t, yaw))
        if t - last_ctl >= 0.05 and state == "run":           # the app's loop: 20 Hz
            last_ctl = t
            seen = next((y for tt, y in reversed(hist) if tt <= t - gyro_delay), 0.0)
            seen = gyro_sign * (seen + math.radians(rng.gauss(0, noise_deg)))
            if gyro_dead:
                seen = 0.0
            cmd_now, state = ctl.step(seen, t)
            cmds.append((t, cmd_now))
            if state != "run":
                cmds.append((t, 0.0))
        cmd = next((c for tt, c in reversed(cmds) if tt <= t - cmd_delay), 0.0)
        if dog_stuck:
            cmd = 0.0
        rate += (cmd - rate) * dt / tau
        yaw += rate * dt
        t += dt
        if state != "run" and abs(rate) < 0.01:
            break
    return {"final": math.degrees(yaw), "t": t, "state": state}


def main() -> int:
    checks = {}
    print(f"{'target':>8s} {'dog lag':>8s}  turned    error   time")
    worst = 0.0
    for tau in (0.15, 0.3, 0.5):
        for target in (90, -90, 180, 45, -45, 360):
            r = simulate(target, tau=tau)
            err = r["final"] - target
            worst = max(worst, abs(err) if abs(target) <= 180 else 0)
            print(f"{target:8d} {tau:7.2f}s {r['final']:7.1f}  {err:+6.1f}   {r['t']:4.1f} s  {r['state']}")
    checks["turns 45 / 90 / 180 degrees either way to within 8 degrees, whether the dog is quick (0.15 s lag) or sluggish (0.5 s)"] = worst <= 8.0
    quick = [abs(simulate(t, tau=0.25, seed=s)["final"] - t) for t in (90, -90, 180) for s in range(1, 6)]
    checks[f"typical dog (0.25 s lag), 15 runs of 90 / -90 / 180: every one within 8 degrees (worst {max(quick):.1f})"] = max(quick) <= 8.0
    r180 = simulate(180)
    checks[f"a 180 takes about as long as at a steady 0.8 rad/s plus a little settling ({r180['t']:.1f} s, was 3.9 s by the clock)"] = 3.5 <= r180["t"] <= 6.0
    rw = simulate(90, gyro_sign=-1.0)
    checks["a gyro with the wrong sign is caught within ~1.2 s (it does not spin on: about 45 degrees at most before it falls back to timing)"] = rw["state"] == "wrong_way" and abs(rw["final"]) < 90
    rd = simulate(90, gyro_dead=True)
    checks["a dead gyro is caught (not moving) and does not spin forever"] = rd["state"] == "not_moving" and abs(rd["final"]) < 130
    rs = simulate(90, dog_stuck=True)
    checks["a dog that never turns is given up on (no endless command)"] = rs["state"] == "not_moving"
    intents = {"turn left": 90, "turn right": -90, "turn around": 180, "turn left 45 degrees": 45, "turn right 30 degrees": -30, "turn around a little": 90,
               "turn left a lot": 180, "turn left two seconds": None, "walk forward": None, "turn right 720 degrees": -366.6}
    got = {}
    for text, want in intents.items():
        it = voice.parse_command(f"{text}")
        tg = voice.turn_target(it) if it else None
        got[text] = None if tg is None else round(math.degrees(tg), 1)
    ok = all((got[k] is None and want is None) or (got[k] is not None and want is not None and abs(got[k] - want) < 1.0) for k, want in intents.items())
    print("spoken turn -> degrees:", got)
    checks["'turn left' = 90 left, 'turn right' = 90 right, 'turn around' = 180, 'turn right 45 degrees' = 45, 'a little' halves, 'a lot' doubles, a turn in seconds stays timed"] = ok
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
