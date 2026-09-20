def edit(p, pairs):
    s = open(p, encoding="utf-8", newline="").read()
    crlf = "\r\n" in s
    s = s.replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\r\n" if crlf else "\n").write(s)


edit("go2.py", [
# real robot: the dog's heading from its IMU
('''    def on_audio(self, cb) -> None:
        """cb(av.AudioFrame) for every audio frame from the dog's own microphone (switches the audio channel on).''',
 '''    def on_yaw(self, cb) -> None:
        """cb(yaw radians, + = counter-clockwise = left, wraps at +-pi) from the dog's own IMU (its low-level state, the same stream as the battery).
        If the message has no IMU heading, cb is simply never called (turns then fall back to timing)."""
        import math

        def handle(msg):
            try:
                imu = msg["data"]["imu_state"]
                rpy = imu.get("rpy")
                if rpy is not None and len(rpy) >= 3:
                    cb(float(rpy[2]))
                    return
                w, x, y, z = (float(v) for v in imu["quaternion"])
                cb(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            except Exception:  # noqa: BLE001 - odd/partial message: just skip it
                pass

        self._subs.append(self.c.lowstate_stream().subscribe(handle))

    def on_audio(self, cb) -> None:
        """cb(av.AudioFrame) for every audio frame from the dog's own microphone (switches the audio channel on).'''),
# fake robot: a simulated gyro that follows what was commanded
('''    def move(self, vx: float, vy: float, yaw: float) -> None:
        self._add(("move", round(vx, 2), round(vy, 2), round(yaw, 2)))

    def stop_move(self) -> None:
        self._add(("stop_move",))

    def on_frame(self, cb) -> None:
        def gen():
            h, w = 720, 1280''', '''    def move(self, vx: float, vy: float, yaw: float) -> None:
        self._add(("move", round(vx, 2), round(vy, 2), round(yaw, 2)))
        self._yaw_cmd, self._yaw_cmd_at = yaw, time.time()

    def stop_move(self) -> None:
        self._add(("stop_move",))
        self._yaw_cmd = 0.0

    def on_yaw(self, cb) -> None:
        """A pretend gyro: the dog's heading, following what was commanded (a little lag, like the real thing)."""
        import math

        def gen():
            yaw, rate, last = 0.0, 0.0, time.time()
            while not self._halt.is_set():
                time.sleep(0.05)
                now = time.time()
                cmd = getattr(self, "_yaw_cmd", 0.0) if now - getattr(self, "_yaw_cmd_at", 0.0) < 0.5 else 0.0
                rate += (cmd - rate) * (now - last) / 0.2
                yaw += rate * (now - last)
                last = now
                cb(math.atan2(math.sin(yaw), math.cos(yaw)))                 # wrapped, like the real one

        threading.Thread(target=gen, daemon=True).start()

    def on_frame(self, cb) -> None:
        def gen():
            h, w = 720, 1280'''),
# app: subscribe with the battery
('''            r.on_battery(lambda soc: setattr(self, "battery", soc))''',
 '''            r.on_battery(lambda soc: setattr(self, "battery", soc))
            try:
                r.on_yaw(self._on_yaw)                          # the dog's own gyro: spoken turns stop at the angle you asked for
            except Exception as e:  # noqa: BLE001
                self.say(f"gyro unavailable ({e}): spoken turns will be timed instead", WARN)'''),
# app state
('''        self.phone = None                   # phonelink.PhoneLink: the iPhone's motion (--phone)''',
 '''        self.yaw_total = 0.0                # the dog's heading in radians, unwrapped (from its IMU); yaw_at = when the last reading came
        self.yaw_at = 0.0
        self._yaw_prev = None
        self.phone = None                   # phonelink.PhoneLink: the iPhone's motion (--phone)'''),
# start_voice_move
('''        cmd, secs = voice_mod.plan_motion(intent, self.args.linear, self.args.angular)
        self.stop_follow("voice move")
        self.stop_lead("voice move")
        self.abort.set()   # a running routine must not keep issuing tricks while we walk
        self.pending = None
        self.voice_move = {"cmd": cmd, "dur": secs, "start": None, "created": time.time(), "label": intent.label}
        self.say(f"> {intent.label} for {secs:.1f} s   (say 'stop' / Space / any key cancels)")''',
 '''        cmd, secs = voice_mod.plan_motion(intent, self.args.linear, self.args.angular)
        target = voice_mod.turn_target(intent)                  # a turn asked for as an angle ("turn left" = 90, "turn around" = 180): the gyro stops it
        self.stop_follow("voice move")
        self.stop_lead("voice move")
        self.abort.set()   # a running routine must not keep issuing tricks while we walk
        self.pending = None
        self.voice_move = {"cmd": cmd, "dur": secs, "start": None, "created": time.time(), "label": intent.label}
        if target is not None:
            self.voice_move.update(target=target, yaw0=None, ctl=voice_mod.TurnController(target, abs(cmd[2])))
            self.say(f"> {intent.label} {abs(math.degrees(target)):.0f} degrees   (say 'stop' / Space / any key cancels)")
        else:
            self.say(f"> {intent.label} for {secs:.1f} s   (say 'stop' / Space / any key cancels)")'''),
# update_velocity: closed loop
('''                if vm["start"] is None:
                    vm["start"] = now
                if now - vm["start"] >= vm["dur"]:
                    self.voice_move = None
                    self.desired = (0.0, 0.0, 0.0)
                    self.say(f"{vm['label']}: done")
                    return
                self.desired = vm["cmd"]
                return''', '''                if vm["start"] is None:
                    vm["start"] = now
                if vm.get("target") is not None:
                    if self._turn_step(vm, now):
                        return
                if now - vm["start"] >= vm["dur"] and vm.get("target") is None:
                    self.voice_move = None
                    self.desired = (0.0, 0.0, 0.0)
                    self.say(f"{vm['label']}: done")
                    return
                self.desired = vm["cmd"]
                return'''),
# methods
('''    def start_voice_move(self, intent) -> None:''', '''    def _on_yaw(self, yaw: float) -> None:
        """A heading reading from the dog's IMU (wraps at +-pi): keep a running total so a turn can be measured across the wrap."""
        if self._yaw_prev is not None:
            self.yaw_total += math.atan2(math.sin(yaw - self._yaw_prev), math.cos(yaw - self._yaw_prev))
        self._yaw_prev, self.yaw_at = yaw, time.time()

    def _turn_step(self, vm: dict, now: float) -> bool:
        """One step of a spoken turn measured by the dog's gyro. True = handled (desired is set, or the turn finished); False = the gyro can't be
        trusted, the turn carries on by time like it used to."""
        fresh = now - self.yaw_at < 0.5
        if vm["yaw0"] is None:
            if fresh:
                vm["yaw0"] = self.yaw_total
                vm["t_gyro"] = now
            elif now - vm["start"] > 1.0:
                self.say("turn: no gyro reading from the dog, turning by time instead", WARN)
                vm["target"] = None
                vm["start"] = now
                return False
            else:
                self.desired = (0.0, 0.0, 0.0)
                return True
        if not fresh:
            self.say("turn: lost the gyro reading, finishing by time", WARN)
            vm["target"] = None
            return False
        turned = self.yaw_total - vm["yaw0"]
        rate, state = vm["ctl"].step(turned, now - vm["t_gyro"])
        if state == "run":
            self.desired = (0.0, 0.0, rate)
            return True
        if state in ("wrong_way", "not_moving"):
            self.say(f"turn: the gyro and the dog disagree ({state.replace('_', ' ')}: turned {math.degrees(turned):+.0f} deg), finishing by time", WARN)
            vm["target"] = None
            return False
        self.voice_move = None
        self.desired = (0.0, 0.0, 0.0)
        self.say(f"{vm['label']}: done (turned {abs(math.degrees(turned)):.0f} degrees" + (", gave up waiting" if state == "timeout" else "") + ")", GOOD)
        return True

    def start_voice_move(self, intent) -> None:'''),
])
print("go2 patched")
