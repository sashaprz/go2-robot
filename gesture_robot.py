#!/usr/bin/env python3
"""Real Go2 hand-gesture controller.

This file is separate from gesture_test.py so the webcam test cannot connect
to the robot by accident.

Mapping:
- peace (victory)    -> walk forward
- thumbs_down        -> walk backward
- open_palm          -> sit
- thumbs_up          -> stand up
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_test import GestureController, action_for_gesture, gesture_from_landmarks


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

    # Use MediaPipe HandLandmarker with lower thresholds for robot camera
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,  # Lower threshold for robot camera
        min_hand_presence_confidence=0.3,   # Lower threshold for robot camera
        min_tracking_confidence=0.3,        # Lower threshold for robot camera
        running_mode=vision.RunningMode.VIDEO
    )

    # Download model if needed
    import os as os_module
    if not os_module.path.exists('hand_landmarker.task'):
        print("Downloading hand landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, "hand_landmarker.task")
        print("Model downloaded.")

    landmarker = vision.HandLandmarker.create_from_options(options)

    print("Connecting to robot...")
    robot = Robot(args.ip, aes_key)
    controller = GestureController(robot, forward_speed=args.forward_speed)

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
    print("Commands: peace = walk forward, thumbs_down = walk backward, open_palm = sit, thumbs_up = stand up")

    frame_timestamp_ms = 0
    try:
        while True:
            frame = robot.get_frame()
            if frame is None:
                robot.stop_move()
                time.sleep(0.01)
                continue

            frame_timestamp_ms += 33  # Approximately 30 FPS

            # Frame from robot is already RGB, use it directly for MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Detect hand landmarks
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Convert to BGR for display (frame from robot is RGB)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            action = "NO COMMAND"

            # Debug output
            if results.hand_landmarks:
                print(f"Hand detected! Number of landmarks: {len(results.hand_landmarks[0])}")
            else:
                print("No hand detected")

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                # Draw hand landmarks manually
                h, w, _ = display_frame.shape
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

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
                    cv2.line(display_frame, start_point, end_point, (255, 255, 255), 2)

                gesture = gesture_from_landmarks(hand_landmarks)
                action = action_for_gesture(gesture)

                # Debug output
                print(f"Detected gesture: {gesture}, Action: {action}")

                if action == "NO COMMAND":
                    robot.stop_move()
                else:
                    controller.decide(gesture)

                cv2.putText(display_frame, f"Gesture: {gesture}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                robot.stop_move()

            cv2.putText(display_frame, action, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Go2 gesture control (robot camera)", display_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()
        landmarker.close()
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
