# SentiSign-STT Setup Guide

## Overview

SentiSign-STT uses a **dual environment architecture** to resolve dependency conflicts between TensorFlow and MediaPipe. The main application runs in Python 3.11 with modern dependencies, while the MediaPipe worker runs in Python 3.10 with legacy MediaPipe 0.10.14.

### Why Two Environments?

**The Problem:**
- TensorFlow 2.20+ requires `protobuf >= 6.x` (currently 6.33.3)
- Legacy MediaPipe 0.10.14 requires `protobuf < 5.x` (specifically 4.25.8)
- These versions cannot coexist in the same Python environment

**The Solution:**
- Isolate MediaPipe 0.10.14 in a separate subprocess environment
- Main app runs modern TensorFlow 2.20 with protobuf 6.x
- Worker runs legacy MediaPipe 0.10.14 with protobuf 4.x
- Communication via stdin/stdout using a binary protocol

## Prerequisites

- **Python 3.10.19** (for MediaPipe worker)
- **Python 3.11.13** (for main application)
- **uv** package manager (for dependency management)
- **macOS** or **Linux** (tested on macOS)

### Install uv

```bash
# Using pip
pip install uv

# Or using curl (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

### Main Environment (.venv)

The main environment runs the ASL detection application with TensorFlow and modern MediaPipe.

**Python Version:** 3.11.13

**Dependencies:**
- tensorflow 2.20.0
- mediapipe 0.10.31 (latest compatible version)
- protobuf 6.33.3
- numpy 2.2.6
- opencv-python 4.12.0.88

**Setup Commands:**

```bash
# Create environment and install dependencies
uv sync

# Verify installation
uv run python -c "import tensorflow, mediapipe; print(f'TF: {tensorflow.__version__}, MP: {mediapipe.__version__}')"
# Expected: TF: 2.20.0, MP: 0.10.31
```

### MediaPipe Worker Environment (mp_env)

The worker environment runs legacy MediaPipe 0.10.14 with Holistic support.

**Python Version:** 3.10.19

**Dependencies:**
- mediapipe 0.10.14 (legacy version)
- protobuf 4.25.8
- numpy 2.2.6
- opencv-python 4.12.0.88

**Setup Commands:**

```bash
# Create Python 3.10 environment
uv venv mp_env --python 3.10

# Install legacy MediaPipe and dependencies
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy

# Verify installation
mp_env/bin/python -c "import mediapipe; print(f'MP: {mediapipe.__version__}')"
# Expected: MP: 0.10.14
```

## Verification

Check both environments are correctly configured:

```bash
# Check main environment (.venv)
uv run python -c "import protobuf; print(f'protobuf: {protobuf.__version__}')"
# Expected: protobuf: 6.33.3

# Check worker environment (mp_env)
mp_env/bin/python -c "import protobuf; print(f'protobuf: {protobuf.__version__}')"
# Expected: protobuf: 4.25.8

# List all packages in .venv
uv pip list --python .venv/bin/python

# List all packages in mp_env
uv pip list --python mp_env/bin/python
```

## Running the Application

### ASL Sign Detection (New System with Worker)

The new detection system uses the MediaPipe worker subprocess for landmark extraction.

```bash
uv run detect_signs.py
```

**Features:**
- Real-time ASL sign recognition from webcam
- Uses 66-landmark TensorFlow Lite model
- Predictions every 3 seconds
- Visual landmark overlay
- Confidence scores

**Controls:**
- Press 'q' to quit

### Original Application (Legacy System)

The original application uses MediaPipe directly in the main environment.

```bash
uv run app.py
```

**Note:** This uses MediaPipe 0.10.31 and may have different behavior than the worker-based system.

### Model Training

```bash
uv run jupyter notebook ASL_Model_Training.ipynb
```

Run the notebook cells to train or retrain the ASL recognition model.

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Process (.venv)                    │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │ Webcam       │─────▶│ SignDetector │─────▶│ TFLite   │ │
│  │ Capture      │      │              │      │ Model    │ │
│  └──────────────┘      │              │      └──────────┘ │
│         │              │              │                    │
│         │              │              │                    │
│         ▼              │              │                    │
│  ┌──────────────┐      │              │                    │
│  │ Frame Data   │      │              │                    │
│  └──────┬───────┘      │              │                    │
│         │              │              │                    │
│         │ stdin/stdout │              │                    │
│         ▼              │              │                    │
│  ┌─────────────────────────────────────┴──────────────────┐ │
│  │         MediaPipeWorker (subprocess)                  │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────┬──────────────────────────────────┘
                         │
                         │ binary protocol
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Worker Process (mp_env)                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │ Frame Data   │─────▶│ MediaPipe    │─────▶│ Landmarks│ │
│  │ (BGR)        │      │ Holistic     │      │ (543×3)  │ │
│  └──────────────┘      └──────────────┘      └─────┬────┘ │
│                                                    │      │
└────────────────────────────────────────────────────┼──────┘
                                                     │
                                                     │ stdin/stdout
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Back to Main Process                     │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │ Landmarks    │─────▶│ Preprocess   │─────▶│ Display  │ │
│  │ (543×3)      │      │ & Predict    │      │ Overlay  │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Binary Communication Protocol

The main app and worker communicate via stdin/stdout using a structured binary protocol.

#### Input Format (Main → Worker)

```
Header: 8 bytes
  ├─ height: 4 bytes (int32, little-endian)
  └─ width:  4 bytes (int32, little-endian)

Frame: N bytes
  └─ N = height × width × 3 (BGR pixel data, uint8)
```

**Example:** For a 640×480 frame:
- Header: 8 bytes
- Frame data: 640 × 480 × 3 = 921,600 bytes
- Total: 921,608 bytes

#### Output Format (Worker → Main)

```
Landmarks: 6,516 bytes
  └─ 543 landmarks × 3 coordinates (x, y, z) × 4 bytes/float32
```

**Array Shape:** (543, 3)
**Data Type:** float32
**Coordinate Range:** 0.0 to 1.0 (normalized)

#### Landmark Layout (Kaggle GISLR Format)

| Landmark Type | Indices  | Count | Description |
|---------------|----------|-------|-------------|
| Face          | 0-467    | 468   | Full face mesh |
| Left Hand     | 468-488  | 21    | Hand keypoints |
| Pose          | 489-521  | 33    | Body pose |
| Right Hand    | 522-542  | 21    | Hand keypoints |
| **Total**     | **0-542**| **543**| **All landmarks** |

### Code Implementation

**Main App Side** (`detect_signs.py`):
- `MediaPipeWorker` class (lines 58-122): Handles subprocess communication
- `extract_landmarks()` method (lines 91-114): Sends frames, receives landmarks
- `SignDetector` class (lines 125-437): Main detection logic

**Worker Side** (`mp_worker.py`):
- `main()` function (lines 56-109): Reads frames, extracts landmarks
- `extract_landmarks()` function (lines 20-53): MediaPipe Holistic processing

## Troubleshooting

### Worker Startup Failures

**Symptoms:**
- `RuntimeError: MediaPipe worker failed to start`
- `FileNotFoundError: MediaPipe environment not found`

**Solutions:**

```bash
# Check if mp_env exists and has Python
ls -la mp_env/bin/python

# Check Python version in mp_env
mp_env/bin/python --version
# Expected: Python 3.10.19

# Check MediaPipe installation in mp_env
mp_env/bin/python -c "import mediapipe; print(mediapipe.__version__)"
# Expected: 0.10.14

# Reinstall if needed
uv pip uninstall --python mp_env/bin/python mediapipe
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy
```

### Incomplete Landmarks Error

**Symptom:** `RuntimeError: Worker returned incomplete landmarks`

**Causes:**
- Worker process crashed
- Frame size mismatch
- Buffer overflow
- Corrupted frame data

**Debug Steps:**

```bash
# Check if worker process is running
ps aux | grep mp_worker.py

# Check for errors in worker stderr (see detect_signs.py:86)
# Worker errors are printed to stderr

# Test worker manually (echo frame data to stdin)
# (advanced: requires constructing binary header + frame)
```

### Process Hangs

**Symptom:** Application freezes during landmark extraction

**Solutions:**

```bash
# Kill stuck worker process
pkill -f mp_worker.py

# Verify process is terminated
ps aux | grep mp_worker.py

# Restart application
uv run detect_signs.py
```

### Performance Issues

**Symptoms:**
- High CPU usage
- Slow frame rate (< 10 FPS)
- Laggy UI

**Optimizations:**

1. **Reduce Camera Resolution:**
   ```python
   # In detect_signs.py, modify camera setup
   cap = cv2.VideoCapture(0)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Add this
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Add this
   ```

2. **Skip Frames Between Predictions:**
   Already implemented (prediction every 3 seconds).

3. **Process Every N-th Frame for Visualization:**
   ```python
   # In detect_signs.py run() method
   frame_count += 1
   if frame_count % 2 == 0:  # Only visualize every 2nd frame
       display_image = self.draw_landmarks(image.copy(), landmarks)
   else:
       display_image = image
   ```

4. **Use GPU Acceleration:**
   ```bash
   # Install TensorFlow with GPU support
   uv pip install --python .venv/bin/python tensorflow[and-cuda]  # Linux
   uv pip install --python .venv/bin/python tensorflow-macos     # macOS
   ```

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'mediapipe'` (in mp_env)

**Solution:**
```bash
# Reinstall MediaPipe in mp_env
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy
```

**Symptom:** `ImportError: cannot import name 'holistic'` (in .venv)

**Solution:**
```bash
# This is expected - .venv uses MediaPipe 0.10.31 which has different API
# Use the worker system instead for holistic detection
```

## Performance Notes

### Timing Breakdown

| Operation                    | Time (approx) | Notes |
|------------------------------|---------------|-------|
| Webcam capture               | 1-2 ms        | cv2.VideoCapture |
| Frame serialization           | 0.5 ms        | numpy.tobytes |
| Subprocess communication      | 1 ms          | stdin/stdout |
| MediaPipe Holistic detection | 30-50 ms      | Most expensive |
| Data deserialization          | 0.3 ms        | numpy.frombuffer |
| Preprocessing                | 1-2 ms        | Resizing, normalization |
| TFLite inference             | 5-10 ms       | Every 3 seconds |
| Drawing landmarks            | 1-2 ms        | OpenCV operations |
| **Total per frame**          | **35-55 ms**  | **~20-28 FPS** |

### Bottlenecks

1. **MediaPipe Holistic:** 30-50ms per frame (70-80% of time)
2. **Subprocess overhead:** ~1.5ms per frame (negligible)
3. **TensorFlow inference:** 5-10ms per prediction (infrequent)

### Optimization Opportunities

1. **Lower Resolution:** 640×480 instead of 1920×1080
   - Reduces MediaPipe time by ~40%
   - Improves overall FPS from ~20 to ~35

2. **Skip Landmark Visualization:** Process every 2nd frame
   - Saves ~3ms per frame
   - Minimal impact on detection

3. **Use Threading:** Parallel capture and processing
   - Potential 20-30% improvement
   - Requires code refactoring

4. **Reduce Prediction Frequency:** 5 seconds instead of 3
   - Fewer TFLite inference calls
   - Better for slower systems

### System Requirements

**Minimum:**
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- OS: macOS 10.15+, Ubuntu 20.04+

**Recommended:**
- CPU: 4+ cores, 2.5+ GHz
- RAM: 8 GB
- OS: macOS 11+, Ubuntu 22.04+
- GPU: NVIDIA GPU with CUDA (Linux) or Apple Silicon (macOS)

## Additional Resources

- **AGENTS.md:** Coding guidelines and project-specific notes
- **README.md:** Project overview and usage instructions
- **ASL_Model_Training.ipynb:** Model training documentation
- **Kaggle GISLR Dataset:** Reference for landmark format

## Support

For issues or questions:
1. Check this guide's Troubleshooting section
2. Review AGENTS.md for coding guidelines
3. Check MediaPipe documentation: https://google.github.io/mediapipe/
4. Check TensorFlow documentation: https://www.tensorflow.org/lite
