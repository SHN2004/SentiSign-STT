"""
ASL Sign Detection Script
Uses the 66-landmark TFLite model for real-time sign recognition.
Uses subprocess with legacy MediaPipe Holistic for landmark extraction.
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import time
import os
import subprocess
import struct

# ==================== CONFIGURATION ====================
INPUT_SIZE = 64          # Max sequence length (frames)
N_COLS = 66              # Number of selected landmarks
N_DIMS = 3               # x, y, z coordinates
PREDICTION_INTERVAL = 3  # Seconds between predictions

# ==================== LANDMARK INDICES ====================
# Lip landmarks (40 points) from MediaPipe face mesh
LIPS_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])

# Hand landmarks (21 points each)
LEFT_HAND_IDXS = np.arange(468, 489)
RIGHT_HAND_IDXS = np.arange(522, 543)

# Pose landmarks (5 key points) - MUST match gislr-training(1).ipynb
POSE_IDXS = np.array([489, 490, 492, 493, 494])

# Combined indices for left and right hand variants
LANDMARK_IDXS_LEFT = np.concatenate((LIPS_IDXS, LEFT_HAND_IDXS, POSE_IDXS))
LANDMARK_IDXS_RIGHT = np.concatenate((LIPS_IDXS, RIGHT_HAND_IDXS, POSE_IDXS))

# Indices within the 66 selected landmarks
LIPS_START, LIPS_END = 0, 40
HAND_START, HAND_END = 40, 61
POSE_START, POSE_END = 61, 66

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


class MediaPipeWorker:
    """Communicates with legacy MediaPipe subprocess for landmark extraction."""

    def __init__(self, worker_path='mp_worker.py', env_path='mp_env'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        python_path = os.path.join(script_dir, env_path, 'bin', 'python')
        worker_script = os.path.join(script_dir, worker_path)

        if not os.path.exists(python_path):
            raise FileNotFoundError(
                f"MediaPipe environment not found at {python_path}. "
                "Run: uv venv mp_env --python 3.10 && mp_env/bin/pip install mediapipe==0.10.14 opencv-python numpy"
            )

        if not os.path.exists(worker_script):
            raise FileNotFoundError(f"Worker script not found at {worker_script}")

        print(f"Starting MediaPipe worker subprocess...")
        self.process = subprocess.Popen(
            [python_path, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait briefly for worker to initialize and check for errors
        time.sleep(1.0)
        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read().decode()
            raise RuntimeError(f"MediaPipe worker failed to start: {stderr_output}")

        print("MediaPipe worker started successfully")

    def extract_landmarks(self, frame_bgr):
        """
        Send frame to worker, receive landmarks.

        Args:
            frame_bgr: BGR image (numpy array)

        Returns:
            landmarks: (543, 3) array of landmarks
        """
        height, width = frame_bgr.shape[:2]

        # Send header (dimensions) + frame data
        header = struct.pack('ii', height, width)
        self.process.stdin.write(header)
        self.process.stdin.write(frame_bgr.tobytes())
        self.process.stdin.flush()

        # Read landmarks (543 * 3 * 4 = 6516 bytes)
        landmark_bytes = self.process.stdout.read(543 * 3 * 4)
        if len(landmark_bytes) < 543 * 3 * 4:
            raise RuntimeError("Worker returned incomplete landmarks")

        landmarks = np.frombuffer(landmark_bytes, dtype=np.float32).reshape(543, 3)
        return landmarks

    def close(self):
        """Terminate worker process."""
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
            print("MediaPipe worker terminated")


class SignDetector:
    def __init__(self, model_path='model_66landmarks.tflite', labels_path='label_mappings.json'):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load label mappings
        with open(labels_path, 'r') as f:
            mappings = json.load(f)
        self.ord2sign = {int(k): v for k, v in mappings['ord2sign'].items()}

        # Initialize MediaPipe worker subprocess
        self.mp_worker = MediaPipeWorker()

        print(f"Model loaded: {model_path}")
        print(f"Labels loaded: {len(self.ord2sign)} signs")

    def draw_landmarks(self, image, landmarks):
        """Draw landmarks on the image using OpenCV."""
        h, w = image.shape[:2]

        # Draw face landmarks (sparse - every 10th point)
        for i in range(0, 468, 10):
            if not np.isnan(landmarks[i, 0]):
                x, y = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        # Draw left hand (indices 468-488)
        left_hand_pts = []
        for i in range(468, 489):
            if not np.isnan(landmarks[i, 0]):
                x, y = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
                left_hand_pts.append((x, y))
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            else:
                left_hand_pts.append(None)

        # Draw left hand connections
        for conn in HAND_CONNECTIONS:
            if left_hand_pts[conn[0]] and left_hand_pts[conn[1]]:
                cv2.line(image, left_hand_pts[conn[0]], left_hand_pts[conn[1]], (255, 0, 0), 2)

        # Draw right hand (indices 522-542)
        right_hand_pts = []
        for i in range(522, 543):
            if not np.isnan(landmarks[i, 0]):
                x, y = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
                right_hand_pts.append((x, y))
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            else:
                right_hand_pts.append(None)

        # Draw right hand connections
        for conn in HAND_CONNECTIONS:
            if right_hand_pts[conn[0]] and right_hand_pts[conn[1]]:
                cv2.line(image, right_hand_pts[conn[0]], right_hand_pts[conn[1]], (0, 0, 255), 2)

        # Draw pose landmarks (indices 489-521) - key points only
        for i in [489, 490, 491, 492, 493, 494, 495, 496]:  # Upper body only
            if i < 522 and not np.isnan(landmarks[i, 0]):
                x, y = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

        return image

    def determine_dominant_hand(self, frames):
        """
        Determine which hand has more valid data across all frames.
        Returns True if right hand is dominant.
        """
        left_valid = np.sum(~np.isnan(frames[:, LEFT_HAND_IDXS, :]))
        right_valid = np.sum(~np.isnan(frames[:, RIGHT_HAND_IDXS, :]))
        return right_valid > left_valid

    def select_landmarks(self, frames, use_right_hand):
        """Select 66 relevant landmarks from 543."""
        if use_right_hand:
            return frames[:, LANDMARK_IDXS_RIGHT, :]
        else:
            return frames[:, LANDMARK_IDXS_LEFT, :]

    def resize_pad_sequence(self, data, target_length=INPUT_SIZE):
        """Resize sequence to target length using interpolation or padding."""
        current_length = len(data)

        if current_length == 0:
            return np.zeros((target_length, N_COLS, N_DIMS), dtype=np.float32)

        if current_length == target_length:
            return data

        if current_length > target_length:
            # Interpolate to downsample
            indices = np.linspace(0, current_length - 1, target_length).astype(int)
            return data[indices]
        else:
            # Pad with zeros
            padded = np.zeros((target_length, N_COLS, N_DIMS), dtype=np.float32)
            padded[:current_length] = data
            return padded

    def normalize_coordinates(self, data):
        """Normalize coordinates relative to pose landmarks."""
        data = data.copy()
        data = np.nan_to_num(data, nan=0.0)

        # Get pose landmarks for normalization reference
        pose_data = data[:, POSE_START:POSE_END, :]

        # Calculate center point from pose landmarks
        valid_mask = np.any(pose_data != 0, axis=-1, keepdims=True)
        if np.any(valid_mask):
            with np.errstate(invalid='ignore'):
                center = np.nanmean(np.where(valid_mask, pose_data, np.nan), axis=1, keepdims=True)
            center = np.nan_to_num(center, nan=0.0)
            data = data - center

        # Scale to [-1, 1] range
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        return data.astype(np.float32)

    def get_frame_indices(self, data):
        """Get indices of non-empty frames for positional encoding."""
        frame_sums = np.sum(np.abs(data), axis=(1, 2))
        non_empty = frame_sums > 0

        idxs = np.zeros(INPUT_SIZE, dtype=np.int32)
        idxs[non_empty] = np.arange(1, np.sum(non_empty) + 1)

        return idxs

    def preprocess(self, raw_frames):
        """
        Full preprocessing pipeline.

        Args:
            raw_frames: List of (543, 3) arrays from MediaPipe

        Returns:
            frames: (1, 64, 66, 3) array
            frame_idxs: (1, 64) array
        """
        if len(raw_frames) == 0:
            return None, None

        # Stack frames: (N, 543, 3)
        frames = np.stack(raw_frames, axis=0)

        # Determine dominant hand
        use_right = self.determine_dominant_hand(frames)

        # Select 66 landmarks
        frames = self.select_landmarks(frames, use_right)

        # Flip x-coordinates if using right hand (for consistency)
        if use_right:
            frames[:, :, 0] = -frames[:, :, 0]

        # Resize to fixed length
        frames = self.resize_pad_sequence(frames, INPUT_SIZE)

        # Normalize
        frames = self.normalize_coordinates(frames)

        # Get frame indices
        frame_idxs = self.get_frame_indices(frames)

        # Add batch dimension
        frames = np.expand_dims(frames, axis=0).astype(np.float32)
        frame_idxs = np.expand_dims(frame_idxs, axis=0).astype(np.int32)

        return frames, frame_idxs

    def predict(self, frames, frame_idxs):
        """Run inference on preprocessed data."""
        self.interpreter.set_tensor(self.input_details[0]['index'], frames)
        self.interpreter.set_tensor(self.input_details[1]['index'], frame_idxs)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred_idx = np.argmax(output[0])
        confidence = output[0][pred_idx]

        return self.ord2sign[pred_idx], confidence

    def cleanup(self):
        """Close worker process."""
        self.mp_worker.close()

    def run(self):
        """Main capture and detection loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Get frame dimensions
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            cap.release()
            return

        frame_height, frame_width = frame.shape[:2]
        display_width = frame_width + 400  # Extra space for text

        # State variables
        raw_frames = []
        start_time = time.time()
        last_prediction_time = start_time
        current_sign = "Waiting..."
        current_confidence = 0.0

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX

        print("\nSign Detection Started")
        print("Press 'q' to quit")
        print("-" * 40)

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                current_time = time.time()
                elapsed = current_time - start_time

                # Extract landmarks using worker subprocess
                landmarks = self.mp_worker.extract_landmarks(image)
                raw_frames.append(landmarks)

                # Make prediction every PREDICTION_INTERVAL seconds
                if current_time - last_prediction_time >= PREDICTION_INTERVAL:
                    if len(raw_frames) > 10:  # Need minimum frames
                        frames, frame_idxs = self.preprocess(raw_frames)
                        if frames is not None:
                            current_sign, current_confidence = self.predict(frames, frame_idxs)
                            print(f"Detected: {current_sign} ({current_confidence:.1%})")

                    raw_frames = []
                    last_prediction_time = current_time

                # Draw landmarks on image
                display_image = self.draw_landmarks(image.copy(), landmarks)

                # Create display with side panel
                display = np.zeros((frame_height, display_width, 3), dtype=np.uint8)
                display[:, :frame_width] = display_image

                # Draw side panel info
                panel_x = frame_width + 20

                cv2.putText(display, "ASL Sign Detection", (panel_x, 40),
                            font, 0.8, (255, 255, 255), 2)
                cv2.putText(display, "-" * 20, (panel_x, 70),
                            font, 0.5, (100, 100, 100), 1)

                # Current prediction
                cv2.putText(display, "Detected Sign:", (panel_x, 120),
                            font, 0.6, (200, 200, 200), 1)
                cv2.putText(display, current_sign, (panel_x, 160),
                            font, 1.2, (0, 255, 0), 2)

                # Confidence
                cv2.putText(display, f"Confidence: {current_confidence:.1%}", (panel_x, 200),
                            font, 0.6, (200, 200, 200), 1)

                # Progress bar for next prediction
                time_to_next = PREDICTION_INTERVAL - (current_time - last_prediction_time)
                progress = 1 - (time_to_next / PREDICTION_INTERVAL)
                bar_width = 200
                bar_height = 20
                cv2.rectangle(display, (panel_x, 240), (panel_x + bar_width, 240 + bar_height),
                              (100, 100, 100), -1)
                cv2.rectangle(display, (panel_x, 240), (panel_x + int(bar_width * progress), 240 + bar_height),
                              (0, 200, 0), -1)
                cv2.putText(display, f"Next prediction: {time_to_next:.1f}s", (panel_x, 285),
                            font, 0.5, (150, 150, 150), 1)

                # Elapsed time
                cv2.putText(display, f"Elapsed: {int(elapsed)}s", (panel_x, 330),
                            font, 0.5, (150, 150, 150), 1)

                # Frames collected
                cv2.putText(display, f"Frames: {len(raw_frames)}", (panel_x, 360),
                            font, 0.5, (150, 150, 150), 1)

                # Instructions
                cv2.putText(display, "Press 'q' to quit", (panel_x, frame_height - 30),
                            font, 0.5, (100, 100, 100), 1)

                cv2.imshow("ASL Sign Detection", display)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
            print("\nDetection stopped.")


if __name__ == "__main__":
    detector = SignDetector()
    detector.run()
