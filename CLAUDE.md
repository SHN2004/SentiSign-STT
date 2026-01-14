# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SignSense (SentiSign-STT) is a real-time American Sign Language (ASL) recognition system using a transformer-based deep learning model. The recommended runtime (`detect_signs.py`) captures video from a webcam, extracts landmarks with a legacy MediaPipe worker subprocess, preprocesses landmarks, and runs inference with a TensorFlow Lite model.

There is also a legacy runtime (`app.py`) that integrates Gemini for sentence generation, but it is not guaranteed to run out-of-the-box with the current dependency setup.

## Commands

### Install Dependencies (recommended path)
```bash
uv sync
```

### Create MediaPipe Worker Environment (first time only)
```bash
uv venv mp_env --python 3.10
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy
```

### Run (recommended)
```bash
uv run detect_signs.py
```

### Run (legacy; not guaranteed)
```bash
uv run app.py
```

### Train / Explore
```bash
uv run jupyter notebook ASL_Model_Training.ipynb
```

## Architecture

### Data Flow Pipeline
1. **Video Capture**: OpenCV captures frames from webcam (`cv2.VideoCapture(0)`)
2. **Landmark Extraction**: `detect_signs.py` sends frames to `mp_worker.py` (running in `mp_env`) for MediaPipe Holistic landmarks
3. **Preprocessing**: select 66 landmarks, choose dominant hand, resize/pad to 64 frames, normalize, compute `frame_idxs`
4. **Inference**: `model_66landmarks.tflite` predicts the sign every few seconds (default: 3)
5. **Display**: OpenCV overlays landmarks and predicted sign/confidence

### Key Data Structures
- **Worker output**: `(543, 3)` float32 array with NaNs for missing landmarks
- **Model inputs**: `frames` shape `(1, 64, 66, 3)` float32 and `frame_idxs` shape `(1, 64)` int32
- **Label mapping**: `label_mappings.json` provides `ord2sign`

### Model Architecture (in notebook)
- Custom transformer with multi-head attention
- Input: 64 frames × 66 landmarks × 3 dimensions (preprocessed from 543 landmarks)
- Separate embeddings for lips (40), left_hand (21), and pose (5) landmarks
- 2 transformer blocks with 512-dimensional embeddings
- Output: 250 ASL sign classes

### Required Files
- `model_66landmarks.tflite` - Trained TensorFlow Lite model for inference
- `label_mappings.json` - Sign index ↔ name mapping
- `mp_worker.py` - MediaPipe worker (runs in `mp_env`)

Legacy-only (not currently present in this repo): `model.tflite`, `train.csv`.

## Environment Variables

```bash
GOOGLE_API_KEY=<your-gemini-api-key>
```

## UI Controls
- `detect_signs.py`: **q** quits
- `app.py` (legacy): **Escape** toggles Gemini sentence generation, **q** quits
