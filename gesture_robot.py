#!/usr/bin/env python3
"""Real Go2 hand-gesture controller.

This file is separate from gesture_test.py so the webcam test cannot connect
to the robot by accident.

Mapping:
- peace              -> walk forward
- thumbs down        -> walk backward
- horizontal hand    -> sit
- flat hand up       -> stand up
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time

import cv2
import mediapipe as mp

from gesture_test import GestureController, action_for_gesture, gesture_from_landmarks

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class Robot:
    def __init__(self, ip: str, aes_key: str):
        from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
        from dimos.robot.unitree.connection import UnitreeWebRTCConnection

        self._topic = RTC_TOPIC
        self._cmd = SPORT_CMD
        self.c = UnitreeWebRTCConnection(ip, aes_128_key=aes_key)
        self._subs = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def on_frame(self, callback):
        """Register callback for video frames from robot's camera."""
        def frame_handler(f):
            arr = f.to_ndarray(format="rgb24")
            with self.frame_lock:
                self.latest_frame = arr
            if callback:
                callback(arr)
        self._subs.append(self.c.raw_video_stream().subscribe(frame_handler))

    def get_frame(self):
        """Get the latest frame from robot's camera."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def sport(self, name: str) -> None:
        coro = self.c.conn.datachannel.pub_sub.publish_request_new(
            self._topic["SPORT_MOD"], {"api_id": self._cmd[name]}
        )
        asyncio.run_coroutine_threadsafe(coro, self.c.loop).result(timeout=8)

    def move(self, vx: float, vy: float, yaw: float) -> None:
        from dimos.msgs.geometry_msgs.Twist import Twist
        from dimos.msgs.geometry_msgs.Vector3 import Vector3

        twist = Twist()
        twist.linear = Vector3(vx, vy, 0)
        twist.angular = Vector3(0, 0, yaw)
        self.c.move(twist)

    def stop_move(self) -> None:
        self.c.stop_movement()

    def close(self) -> None:
        self.stop_move()
        self.c.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hand gestures on a real Unitree Go2.")
    parser.add_argument("--confirm-real", action="store_true", help="confirm that the real robot may move")
    parser.add_argument("--ip", default="192.168.12.1", help="robot IP")
    parser.add_argument("--forward-speed", type=float, default=0.2, help="forward speed in m/s")
    args = parser.parse_args()

    if not args.confirm_real:
        print("Safety stop: add --confirm-real only when the area around the robot is clear.")
        return 1

    aes_key = os.environ.get("UNITREE_AES_128_KEY")
    if not aes_key:
        print("Missing UNITREE_AES_128_KEY. Load it from ~/.dimos.env first.")
        return 1

    print("Connecting to robot...")
    robot = Robot(args.ip, aes_key)
    controller = GestureController(robot, forward_speed=args.forward_speed)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

    # Start receiving video from robot's camera
    robot.on_frame(None)

    # Wait for first frame
    print("Waiting for robot camera feed...")
    for _ in range(50):  # Wait up to 5 seconds
        if robot.get_frame() is not None:
            break
        time.sleep(0.1)

    if robot.get_frame() is None:
        robot.close()
        print("Could not receive video from robot camera.")
        return 1

    print("Connected. Keep the robot area clear. Press ESC to stop.")
    print("Commands: peace = walk forward, thumbs down = walk backward, horizontal hand = sit, flat hand up = stand up")

    try:
        while True:
            frame = robot.get_frame()
            if frame is None:
                robot.stop_move()
                time.sleep(0.01)
                continue

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            action = "NO COMMAND"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                gesture = gesture_from_landmarks(hand.landmark)
                action = action_for_gesture(gesture)
                if action == "NO COMMAND":
                    robot.stop_move()
                else:
                    controller.decide(gesture)
            else:
                robot.stop_move()

            cv2.putText(frame, action, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Go2 gesture control (robot camera)", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())