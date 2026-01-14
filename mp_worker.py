#!/usr/bin/env python
"""
MediaPipe Holistic landmark extraction worker.
Runs in separate environment with legacy mediapipe (0.10.14).
Communicates via stdin/stdout using binary protocol.

Protocol:
- Input: 4 bytes (height) + 4 bytes (width) + (height*width*3) bytes (BGR frame)
- Output: (543*3*4) bytes (float32 landmarks array)
"""
import sys
import struct
import numpy as np
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def extract_landmarks(frame_rgb, holistic):
    """Extract 543 landmarks from a single frame.

    Landmark layout (matches Kaggle GISLR dataset):
    - Face: indices 0-467 (468 points)
    - Left hand: indices 468-488 (21 points)
    - Pose: indices 489-521 (33 points)
    - Right hand: indices 522-542 (21 points)
    """
    results = holistic.process(frame_rgb)

    landmarks = np.full((543, 3), np.nan, dtype=np.float32)

    # Face landmarks (0-467)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]

    # Left hand landmarks (468-488)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[468 + i] = [lm.x, lm.y, lm.z]

    # Pose landmarks (489-521)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[489 + i] = [lm.x, lm.y, lm.z]

    # Right hand landmarks (522-542)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[522 + i] = [lm.x, lm.y, lm.z]

    return landmarks


def main():
    """Main worker loop - read frames, output landmarks."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    stderr = sys.stderr

    stderr.write("MediaPipe worker starting...\n")
    stderr.flush()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        stderr.write("Holistic model loaded, ready for frames\n")
        stderr.flush()

        while True:
            try:
                # Read frame dimensions (8 bytes: height + width as int32)
                header = stdin.read(8)
                if len(header) < 8:
                    stderr.write("End of input stream\n")
                    break

                height, width = struct.unpack('<ii', header)

                # Read frame data
                frame_size = height * width * 3
                frame_bytes = stdin.read(frame_size)
                if len(frame_bytes) < frame_size:
                    stderr.write(f"Incomplete frame: got {len(frame_bytes)}, expected {frame_size}\n")
                    break

                # Decode frame
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract landmarks
                landmarks = extract_landmarks(frame_rgb, holistic)

                # Write landmarks (543 * 3 * 4 = 6516 bytes)
                stdout.write(landmarks.tobytes())
                stdout.flush()

            except Exception as e:
                stderr.write(f"Worker error: {e}\n")
                stderr.flush()
                break

    stderr.write("MediaPipe worker exiting\n")


if __name__ == "__main__":
    main()
