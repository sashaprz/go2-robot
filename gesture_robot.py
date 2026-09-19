#!/usr/bin/env python3
"""Real Go2 hand-gesture controller.

This file is separate from gesture_test.py so the webcam test cannot connect
to the robot by accident.

Mapping:
- peace       -> walk forward
- thumbs down -> walk backward
- fist        -> sit
"""

from __future__ import annotations

import argparse
import asyncio
import os

import cv2
import mediapipe as mp

from gesture_test import GestureController, action_for_gesture, gesture_from_landmarks

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class Robot:
    def __init__(self, ip: str, aes_key: str):
        from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
        from dimos.robot.unitree.connection import UnitreeWebRTCConnection

        self.topic = RTC_TOPIC
        self.command = SPORT_CMD
        self.connection = UnitreeWebRTCConnection(ip, aes_128_key=aes_key)

    def sport(self, name: str) -> None:
        request = self.connection.conn.datachannel.pub_sub.publish_request_new(
            self.topic["SPORT_MOD"], {"api_id": self.command[name]}
        )
        asyncio.run_coroutine_threadsafe(request, self.connection.loop).result(timeout=8)

    def move(self, vx: float, vy: float, yaw: float) -> None:
        from dimos.msgs.geometry_msgs.Twist import Twist
        from dimos.msgs.geometry_msgs.Vector3 import Vector3

        twist = Twist()
        twist.linear = Vector3(vx, vy, 0)
        twist.angular = Vector3(0, 0, yaw)
        self.connection.move(twist)

    def stop_move(self) -> None:
        self.connection.stop_movement()

    def close(self) -> None:
        self.stop_move()
        self.connection.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hand gestures on a real Unitree Go2.")
    parser.add_argument("--confirm-real", action="store_true", help="confirm that the real robot may move")
    parser.add_argument("--ip", default="192.168.12.1", help="robot IP")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index")
    parser.add_argument("--forward-speed", type=float, default=0.2, help="forward speed in m/s")
    args = parser.parse_args()

    if not args.confirm_real:
        print("Safety stop: add --confirm-real only when the area around the robot is clear.")
        return 1

    aes_key = os.environ.get("UNITREE_AES_128_KEY")
    if not aes_key:
        print("Missing UNITREE_AES_128_KEY. Load it from ~/.dimos.env first.")
        return 1

    robot = Robot(args.ip, aes_key)
    controller = GestureController(robot, forward_speed=args.forward_speed)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    camera = cv2.VideoCapture(args.camera_index)

    if not camera.isOpened():
        robot.close()
        print(f"Could not open camera index {args.camera_index}.")
        return 1

    print("Connected. Keep the robot area clear. Press ESC to stop.")
    print("Commands: peace = walk forward, thumbs down = walk backward, fist = sit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                robot.stop_move()
                break

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
            cv2.imshow("Go2 real gesture control", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())