#!/usr/bin/env python3
"""Beginner hand-gesture controller for the Go2 robot.

This is a webcam-only test:
- detect a hand with MediaPipe
- classify a few easy gestures
- print the matching action without connecting to a robot

Useful beginner mapping:
- peace              -> walk forward
- thumbs down        -> walk backward
- horizontal hand    -> sit
- flat hand up       -> stand up

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
        "horizontal_hand": "SIT",
        "flat_hand": "STAND UP",
    }.get(gesture, "NO COMMAND")


def gesture_from_landmarks(landmarks):
    """Very simple gesture detector for a beginner setup."""
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    # More robust finger detection: check if finger is extended by comparing distances
    # Extended finger: tip is far from MCP joint
    # Curled finger: tip is close to MCP joint
    def finger_extended(tip, mcp, pip):
        tip_to_mcp = math.dist((tip.x, tip.y), (mcp.x, mcp.y))
        pip_to_mcp = math.dist((pip.x, pip.y), (mcp.x, mcp.y))
        # If tip is at least 1.5x further from MCP than PIP is, finger is extended
        return tip_to_mcp > pip_to_mcp * 1.5

    index_open = finger_extended(index_tip, index_mcp, index_pip)
    middle_open = finger_extended(middle_tip, middle_mcp, middle_pip)
    ring_open = finger_extended(ring_tip, ring_mcp, ring_pip)
    pinky_open = finger_extended(pinky_tip, pinky_mcp, pinky_pip)

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

    # All fingers extended: check orientation
    if index_open and middle_open and ring_open and pinky_open:
        # Calculate average horizontal vs vertical spread of fingers
        avg_x_dist = abs(index_tip.x - pinky_tip.x)
        avg_y_dist = abs(index_tip.y - pinky_tip.y)

        # Horizontal hand (karate chop): fingers spread more horizontally than vertically
        if avg_x_dist > avg_y_dist * 1.3:
            return "horizontal_hand"

        # Palm down: fingers pointing down (tips below PIPs in screen coordinates)
        fingers_down = (index_tip.y > index_pip.y and
                       middle_tip.y > middle_pip.y and
                       ring_tip.y > ring_pip.y and
                       pinky_tip.y > pinky_pip.y)

        if fingers_down:
            return "palm_down"
        else:
            # Flat hand / palm forward: fingers pointing up
            return "flat_hand"

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
        elif gesture == "horizontal_hand":
            print("SIT")
            if hasattr(self.robot, "sport"):
                self.robot.sport("Sit")
            else:
                self.robot.stop_move()
        elif gesture == "flat_hand":
            print("STAND UP")
            if hasattr(self.robot, "sport"):
                self.robot.sport("StandUp")
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
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try another index.")
        return 1

    print("Press ESC to quit.")
    print("Commands: peace = walk forward, thumbs down = walk backward, horizontal hand = sit, flat hand up = stand up")

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

                # Show detected gesture and finger states for debugging
                def finger_ext_debug(tip_idx, mcp_idx, pip_idx):
                    tip = hand.landmark[tip_idx]
                    mcp = hand.landmark[mcp_idx]
                    pip = hand.landmark[pip_idx]
                    tip_to_mcp = math.dist((tip.x, tip.y), (mcp.x, mcp.y))
                    pip_to_mcp = math.dist((pip.x, pip.y), (mcp.x, mcp.y))
                    return tip_to_mcp > pip_to_mcp * 1.5

                index_open = finger_ext_debug(8, 5, 6)
                middle_open = finger_ext_debug(12, 9, 10)
                ring_open = finger_ext_debug(16, 13, 14)
                pinky_open = finger_ext_debug(20, 17, 18)
                debug = f"I:{int(index_open)} M:{int(middle_open)} R:{int(ring_open)} P:{int(pinky_open)}"

                cv2.putText(frame, action, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"gesture: {gesture}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, debug, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Go2 gesture control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
