#!/usr/bin/env python3
"""Beginner hand-gesture controller for the Go2 robot.

This is a webcam-only test:
- detect a hand with MediaPipe
- classify a few easy gestures
- print the matching action without connecting to a robot

Useful beginner mapping:
- peace       -> walk forward
- thumbs down -> walk backward
- fist        -> sit

Run:
    py gesture_test.py
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class FakeRobot:
    """Simple stand-in for testing without a real robot."""

    def __init__(self):
        self.log = []

    def sport(self, name: str) -> None:
        self.log.append(("sport", name))
        print(f"FAKE SPORT: {name}")

    def move(self, vx: float, vy: float, yaw: float) -> None:
        self.log.append(("move", round(vx, 2), round(vy, 2), round(yaw, 2)))
        print(f"FAKE MOVE: vx={vx}, vy={vy}, yaw={yaw}")

    def stop_move(self) -> None:
        self.log.append(("stop_move",))
        print("FAKE STOP")


def action_for_gesture(gesture: str) -> str:
    return {
        "peace": "WALK FORWARD",
        "thumbs_down": "WALK BACKWARD",
        "fist": "SIT",
    }.get(gesture, "NO COMMAND")


def gesture_from_landmarks(landmarks):
    """Very simple gesture detector for a beginner setup."""
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    index_open = index_tip.y < index_pip.y
    middle_open = middle_tip.y < middle_pip.y
    ring_open = ring_tip.y < ring_pip.y
    pinky_open = pinky_tip.y < pinky_pip.y

    fingers_closed = not index_open and not middle_open and not ring_open and not pinky_open
    thumb_extended = math.dist((thumb_tip.x, thumb_tip.y), (thumb_mcp.x, thumb_mcp.y)) > (
        math.dist((thumb_ip.x, thumb_ip.y), (thumb_mcp.x, thumb_mcp.y)) * 1.25
    )

    if fingers_closed:
        if thumb_extended and thumb_tip.y < thumb_ip.y - 0.03:
            return "thumbs_up"
        if thumb_extended and thumb_tip.y > thumb_ip.y + 0.03:
            return "thumbs_down"
        if not thumb_extended:
            return "fist"
        return "unknown"

    if index_open and middle_open and not ring_open and not pinky_open:
        return "peace"

    return "unknown"


class GestureController:
    def __init__(self, robot, forward_speed: float = 0.4, turn_speed: float = 0.8):
        self.robot = robot
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.last_action = None
        self.last_action_time = 0.0
        self.last_gesture = None
        self.last_gesture_time = 0.0

    def _send_action(self, gesture: str) -> None:
        if gesture == "peace":
            print("WALK FORWARD")
            self.robot.move(self.forward_speed, 0.0, 0.0)
        elif gesture == "thumbs_down":
            print("WALK BACKWARD")
            self.robot.move(-self.forward_speed, 0.0, 0.0)
        elif gesture == "fist":
            print("SIT")
            if hasattr(self.robot, "sport"):
                self.robot.sport("Sit")
            else:
                self.robot.stop_move()
        else:
            return

    def decide(self, gesture: str) -> None:
        now = time.time()

        if gesture == "unknown":
            self.last_gesture = "unknown"
            self.last_gesture_time = now
            return

        if self.last_gesture != gesture:
            self.last_gesture = gesture
            self.last_gesture_time = now
            return

        if now - self.last_gesture_time < 0.25:
            return

        if self.last_action == gesture and now - self.last_action_time < 0.75:
            return

        self.last_action = gesture
        self.last_action_time = now
        self._send_action(gesture)


def main() -> int:
    parser = argparse.ArgumentParser(description="Beginner hand gesture controller for the Go2 robot.")
    parser.add_argument("--forward-speed", type=float, default=0.4, help="forward speed in m/s")
    parser.add_argument("--turn-speed", type=float, default=0.8, help="turn speed in rad/s")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index, usually 0 or 1")
    args = parser.parse_args()

    robot = FakeRobot()
    print("Test mode: no real robot connection")
    controller = GestureController(robot, forward_speed=args.forward_speed, turn_speed=args.turn_speed)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try another index.")
        return 1

    print("Press ESC to quit.")
    print("Commands: peace = walk forward, thumbs down = walk backward, fist = sit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                gesture = gesture_from_landmarks(hand.landmark)
                controller.decide(gesture)
                action = action_for_gesture(gesture)
                cv2.putText(frame, action, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Go2 gesture control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
