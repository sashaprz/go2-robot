#!/usr/bin/env python3
"""Beginner hand-gesture controller for the Go2 robot.

This is a webcam-only test:
- detect hand gestures with MediaPipe Hands + improved angle-based classifier
- print the matching action without connecting to a robot

Useful beginner mapping:
- peace (victory)    -> walk forward
- thumbs_down        -> walk backward
- open_palm          -> sit
- thumbs_up          -> stand up

Run:
    python3 gesture_test.py
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
    """Map gesture names to robot actions."""
    return {
        "peace": "WALK FORWARD",
        "thumbs_down": "WALK BACKWARD",
        "open_palm": "SIT",
        "thumbs_up": "STAND UP",
    }.get(gesture, "NO COMMAND")


def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c."""
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians))
    return angle if angle <= 180 else 360 - angle


def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx, wrist_idx):
    """Check if a finger is extended using angle-based detection."""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]

    # Distance from tip to wrist vs pip to wrist
    tip_dist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
    pip_dist = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)

    # Finger is extended if tip is significantly further from wrist than pip
    return tip_dist > pip_dist * 1.2


def is_thumb_up(landmarks):
    """Detect thumbs up gesture."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]

    # Thumb pointing up (tip above ip)
    thumb_pointing_up = thumb_tip.y < thumb_ip.y - 0.05

    # Other fingers curled
    index_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    middle_ext = is_finger_extended(landmarks, 12, 10, 9, 0)
    ring_ext = is_finger_extended(landmarks, 16, 14, 13, 0)
    pinky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)

    return thumb_pointing_up and not index_ext and not middle_ext and not ring_ext and not pinky_ext


def is_thumb_down(landmarks):
    """Detect thumbs down gesture."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    # Thumb pointing down (tip below ip)
    thumb_pointing_down = thumb_tip.y > thumb_ip.y + 0.05

    # Other fingers curled
    index_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    middle_ext = is_finger_extended(landmarks, 12, 10, 9, 0)
    ring_ext = is_finger_extended(landmarks, 16, 14, 13, 0)
    pinky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)

    return thumb_pointing_down and not index_ext and not middle_ext and not ring_ext and not pinky_ext


def gesture_from_landmarks(landmarks):
    """Improved gesture classifier using angles and distances."""
    wrist = landmarks[0]

    # Check finger states
    index_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    middle_ext = is_finger_extended(landmarks, 12, 10, 9, 0)
    ring_ext = is_finger_extended(landmarks, 16, 14, 13, 0)
    pinky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)

    # Check thumb gestures first (most distinctive)
    if is_thumb_up(landmarks):
        return "thumbs_up"

    if is_thumb_down(landmarks):
        return "thumbs_down"

    # Peace sign: index and middle extended, ring and pinky curled
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"

    # Open palm: all fingers extended
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open_palm"

    # Fist: all fingers curled
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "fist"

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
        elif gesture == "open_palm":
            print("SIT")
            if hasattr(self.robot, "sport"):
                self.robot.sport("Sit")
            else:
                self.robot.stop_move()
        elif gesture == "thumbs_up":
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

    # Use MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO
    )

    # Download model if needed
    import os
    if not os.path.exists('hand_landmarker.task'):
        print("Downloading hand landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, "hand_landmarker.task")
        print("Model downloaded.")

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try another index.")
        return 1

    print("Press ESC to quit.")
    print("Commands: peace = walk forward, thumbs_down = walk backward, open_palm = sit, thumbs_up = stand up")

    frame_timestamp_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_timestamp_ms += 33  # Approximately 30 FPS

        # Convert to RGB and create MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hand landmarks
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw hand landmarks manually
                h, w, _ = frame.shape
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]
                for connection in connections:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

                gesture = gesture_from_landmarks(hand_landmarks)
                controller.decide(gesture)
                action = action_for_gesture(gesture)

                # Display results
                cv2.putText(frame, action, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {gesture}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Go2 gesture control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
