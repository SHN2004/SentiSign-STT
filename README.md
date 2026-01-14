# SentiSign-STT: Real-time ASL Sign Detection

Real-time American Sign Language (ASL) recognition system using deep learning and computer vision. The system detects ASL signs from live webcam input and translates them into text, supporting communication accessibility.

## Features

- **Real-time ASL Recognition**: Detect and recognize ASL gestures from a live webcam feed
- **Dual Environment Architecture**: Isolates TensorFlow and MediaPipe dependencies to resolve version conflicts
- **TensorFlow Lite Model**: Fast, efficient inference on CPU
- **MediaPipe Holistic**: Accurate landmark extraction for hands, face, and pose
- **Live Webcam Support**: Seamless real-time sign translation
- **66-Landmark Model**: Optimized model using key facial, hand, and pose landmarks
- **Confidence Scoring**: Displays prediction confidence for each detected sign

## Project Overview

SentiSign-STT uses a transformer-based deep learning model trained on the Kaggle Google Isolated Sign Language Recognition (GISLR) dataset. The system processes video frames, extracts landmarks, and classifies ASL signs using a TensorFlow Lite model.

### Dual Environment Architecture

**The Problem:**
- TensorFlow 2.20+ requires `protobuf >= 6.x` (6.x series)
- Legacy MediaPipe 0.10.14 requires `protobuf < 5.x` (4.x series)
- These versions are incompatible in the same environment

**The Solution:**
- **Main Environment (.venv)**: Runs Python 3.11+ with TensorFlow and modern MediaPipe (uv-managed)
- **Worker Environment (mp_env)**: Runs Python 3.10 with MediaPipe 0.10.14 for Holistic detection
- **Communication**: Subprocess communicates via stdin/stdout using a binary protocol
- **Isolation**: Complete dependency separation prevents version conflicts

## Tech Stack

### Core Technologies
- **Python 3.11+** (main application)
- **Python 3.10** (MediaPipe worker)
- **TensorFlow 2.20+**: Deep learning framework
- **MediaPipe 0.10.31+**: Modern version (main environment)
- **MediaPipe 0.10.14**: Legacy version with Holistic support (worker)
- **OpenCV**: Computer vision and video capture
- **NumPy**: Numerical computing

### Package Management
- **uv**: Fast Python package manager for dependency management

## Quick Start

### Prerequisites

- **Python 3.10** and **Python 3.11+**
- **uv** package manager
- **macOS** or **Linux** (tested on macOS)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DEV-D-GR8/SignSense.git
   cd SignSense
   ```

2. **Install uv** (if not already installed):
   ```bash
   pip install uv
   # or
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Setup the main environment:**
   ```bash
   uv sync
   ```

4. **Setup the MediaPipe worker environment:**
   ```bash
   uv venv mp_env --python 3.10
   uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy
   ```

5. **Verify installations:**
   ```bash
   # Check main environment
   uv run python -c "import tensorflow, mediapipe; print(f'TF: {tensorflow.__version__}, MP: {mediapipe.__version__}')"

   # Check worker environment
   mp_env/bin/python -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')"
   # Expected: MediaPipe: 0.10.14

   # Check protobuf versions (main vs worker)
   uv run python -c "import google.protobuf; print(f'protobuf: {google.protobuf.__version__}')"
   mp_env/bin/python -c "import google.protobuf; print(f'protobuf: {google.protobuf.__version__}')"
   ```

### Running the Application

**New Detection System (Recommended):**
```bash
uv run detect_signs.py
```

**Sentence generation (Ollama, local):**
```bash
# Uses OLLAMA_MODEL if set, otherwise defaults to granite3.3:2b
uv run detect_signs_sentences.py
```
Keys: `a` add word, `s` generate, `b` backspace, `c` clear, `q` quit.
Select sentence: arrow keys (or `j`/`k` / `1`–`3`).

**Optional MediaPipe tuning (worker):**
```bash
MP_MIN_DET_CONF=0.6 MP_MIN_TRACK_CONF=0.6 uv run detect_signs.py
```

**Legacy System:**
```bash
uv run app.py
```

**Note:** `app.py` is a legacy pipeline and may not run out-of-the-box with the current `pyproject.toml` dependencies. It also expects additional files (e.g. `model.tflite`, `train.csv`) that are not currently present in this repo.

**Train Model:**
```bash
uv run jupyter notebook ASL_Model_Training.ipynb
```

## Project Structure

```
SentiSign-STT/
├── detect_signs.py          # Main ASL detection app with worker subprocess
├── mp_worker.py             # MediaPipe Holistic worker (runs in mp_env)
├── app.py                   # Legacy application (Gemini + Parquet pipeline)
├── model_66landmarks.tflite # TensorFlow Lite model (66 landmarks)
├── label_mappings.json      # Sign name mappings
├── pyproject.toml           # uv dependency configuration
├── AGENTS.md                # Coding guidelines and developer documentation
├── SETUP.md                 # Detailed setup and troubleshooting guide
└── README.md                # This file

Virtual Environments:
├── .venv/                  # Main environment (Python 3.11+, TensorFlow 2.20+)
└── mp_env/                 # Worker environment (Python 3.10, MediaPipe 0.10.14)

Training:
├── ASL_Model_Training.ipynb     # Model training notebook
├── gislr-training.ipynb         # GISLR dataset training
├── kaggle_training_notebook.ipynb # Kaggle training reference
└── best_model.keras             # Trained Keras model (ignored by git)
```

## Architecture

### System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                     Main Process (.venv)                    │
│  Python 3.11+ │ TensorFlow 2.20+ │ MediaPipe 0.10.31+       │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐  │
│  │ Webcam       │─────▶│ SignDetector │─────▶│ TFLite    │  │
│  │ Capture      │      │              │      │ Model     │  │
│  └──────────────┘      └───────┬──────┘      └───────────┘  │
│         │                      │                            │
│         ▼                      ▼                            │
│  ┌──────────────┐     ┌────────────────┐                    │
│  │ Frame Data   │────▶│ MediaPipeWorker│                    │
│  └──────────────┘     │ (subprocess    │                    │
│                       │ wrapper)       │                    │
│                       └────────┬───────┘                    │
│                                │                            │
└────────────────────────────────┼────────────────────────────┘
                                 │ stdin/stdout (binary protocol)
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Worker Process (mp_env)                  │
│  Python 3.10.x │ MediaPipe 0.10.14 │ protobuf 4.x           │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐  │
│  │ Frame Data   │─────▶│ MediaPipe    │─────▶│ Landmarks │  │
│  │ (BGR)        │      │ Holistic     │      │ (543×3)   │  │
│  └──────────────┘      └──────────────┘      └─────┬─────┘  │
└────────────────────────────────────────────────────┼────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│               Landmarks back to Main Process                │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐  │
│  │ Landmarks    │─────▶│ Preprocess   │─────▶│ Display   │  │
│  │ (543×3)      │      │ & Predict    │      │ Overlay   │  │
│  └──────────────┘      └──────────────┘      └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Binary Communication Protocol

**Main → Worker:**
```
Header: 8 bytes
  - height: 4 bytes (int32)
  - width: 4 bytes (int32)

Frame: N bytes
  - N = height × width × 3 (BGR pixel data, uint8)
```

**Worker → Main:**
```
Landmarks: 6,516 bytes
  - 543 landmarks × 3 coordinates (x, y, z) × 4 bytes/float32
```

### Landmark Layout

| Landmark Type | Indices  | Count | Description |
|---------------|----------|-------|-------------|
| Face          | 0-467    | 468   | Full face mesh |
| Left Hand     | 468-488  | 21    | Hand keypoints |
| Pose          | 489-521  | 33    | Body pose |
| Right Hand    | 522-542  | 21    | Hand keypoints |
| **Total**     | **0-542**| **543**| **All landmarks** |

## Usage

### Controls

- `detect_signs.py`: **q** quits
- `app.py` (legacy): **ESC** toggles Gemini sentence display, **q** quits

### Detection Loop

1. **Capture Frame**: Webcam captures video frame
2. **Send to Worker**: Frame sent via stdin to mp_worker.py
3. **Extract Landmarks**: MediaPipe Holistic extracts 543 landmarks
4. **Return Landmarks**: Worker returns landmarks via stdout
5. **Preprocess**: Select 66 relevant landmarks, normalize, pad
6. **Predict**: TensorFlow Lite model classifies sign every 3 seconds
7. **Display**: Show webcam with landmark overlay and prediction

### Model Performance

- **Frame Rate**: depends on CPU/camera resolution
- **Prediction Interval**: Every 3 seconds
- **Confidence Threshold**: Displays confidence for each prediction
- **Model Size**: 2.6 MB (TensorFlow Lite)

## Troubleshooting

### Common Setup Issues

**Issue**: `FileNotFoundError: MediaPipe environment not found`
- **Solution**: Ensure mp_env exists: `ls -la mp_env/bin/python`
- **Setup**: Run `uv venv mp_env --python 3.10` and install dependencies

**Issue**: `RuntimeError: MediaPipe worker failed to start`
- **Cause**: Worker crashed during initialization
- **Solution**: Check mp_env has mediapipe: `mp_env/bin/python -c "import mediapipe"`

**Issue**: `RuntimeError: Worker returned incomplete landmarks`
- **Cause**: Worker process crashed or frame size mismatch
- **Solution**: Check if worker is running: `ps aux | grep mp_worker.py`

### Performance Issues

**High CPU usage, slow frame rate:**
- Reduce camera resolution in `detect_signs.py`
- Skip frames between predictions
- Close other applications

**Application freezes:**
- Kill stuck worker: `pkill -f mp_worker.py`
- Restart application

For detailed troubleshooting, see [SETUP.md](SETUP.md).

## System Requirements

### Minimum
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **OS**: macOS 10.15+, Ubuntu 20.04+
- **Python**: 3.10 and 3.11+

### Recommended
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8 GB
- **OS**: macOS 11+, Ubuntu 22.04+
- **GPU**: Apple Silicon (macOS) or NVIDIA with CUDA (Linux)

## Model Training

The model is trained on the Kaggle Google Isolated Sign Language Recognition (GISLR) dataset.

**Training Notebooks:**
- `ASL_Model_Training.ipynb`: Main training pipeline
- `gislr-training.ipynb`: GISLR dataset training
- `kaggle_training_notebook.ipynb`: Reference training code

**Model Details:**
- **Architecture**: Transformer-based
- **Input**: 64 frames × 66 landmarks × 3 coordinates (x, y, z)
- **Output**: 250 ASL sign classes
- **Training**: Pre-trained on GISLR dataset

## Additional Documentation

- **[SETUP.md](SETUP.md)**: Detailed setup guide, architecture, and troubleshooting
- **[AGENTS.md](AGENTS.md)**: Coding guidelines, conventions, and developer docs

## Demo

For a visual demonstration, check out the [YouTube video](https://youtu.be/6XNY6YBXgyI?si=RoZdn_8jL35EMuYD).

## Contributing

Contributions are welcome! Please read the existing documentation and follow the coding guidelines in AGENTS.md.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for more information.

## Acknowledgments

- Kaggle GISLR Dataset for training data
- Google MediaPipe for landmark extraction
- TensorFlow for deep learning framework
- Gemini-Pro LLM API for sentence construction (legacy app)
