# AGENTS.md

This file provides guidelines for agentic coding assistants working in this repository.

## Commands

```bash
# Initial setup (first time only)
uv sync

# Create and setup MediaPipe worker environment (first time only)
uv venv mp_env --python 3.10
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy

# Run ASL sign detection (new system with worker)
uv run detect_signs.py

# Run the original application
uv run app.py

# Train model (run cells in Jupyter notebook)
uv run jupyter notebook ASL_Model_Training.ipynb
```

**Testing**: No test framework exists. Test manually by running the application.

## Code Style Guidelines

### Import Order
Standard library → third-party → local. See app.py:1-9 for example.

### Naming Conventions
- Functions/variables: `snake_case` (e.g., `do_capture_loop`, `sign_name`)
- Constants: `ALL_CAPS` (e.g., `ROWS_PER_FRAME`)
- Classes: `PascalCase`
- Mappings: Descriptive suffixes (`SIGN2ORD`, `ORD2SIGN`)

### Error Handling
Check camera capture success before processing (app.py:106-109). Validate API keys before calls. Handle file I/O failures gracefully.

```python
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    return
```

### Environment Variables
Store sensitive data in `.env` file. Load with `load_dotenv()`. The `GOOGLE_API_KEY` is required for Gemini-Pro API.

### API Keys
Never commit `.env` files. Use `os.getenv("GOOGLE_API_KEY")` for the Gemini-Pro API key.

### Documentation
Minimal comments. Add docstrings only for complex functions.

### Constants
Define at module level: `ROWS_PER_FRAME = 543`, `data_columns = ['x', 'y', 'z']`

### Data Handling
Use pandas for tabular data. Parquet format for landmark storage (`pd.read_parquet`, `to_parquet`). NumPy arrays for numerical operations.

### TensorFlow
```python
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
pred_fn = interpreter.get_signature_runner("serving_default")
```

### MediaPipe
```python
with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
    results = holistic.process(image)
```

### OpenCV
```python
cap = cv2.VideoCapture(0)
# ... processing ...
cap.release()
cv2.destroyAllWindows()
```

### Pandas
```python
pd.concat([df1, df2]).reset_index(drop=True)
df.assign(new_col=value)
```

## Dual Environment Architecture

### Why Separate Environments?

**Protobuf Version Conflict:**
- TensorFlow 2.20.0 requires `protobuf >= 6.x` (specifically 6.33.3)
- MediaPipe 0.10.14 requires `protobuf < 5.x` (specifically 4.25.8)
- These versions are incompatible in the same environment

### Environment Breakdown

**`.venv` (Main Environment - Python 3.11.13)**
- tensorflow 2.20.0
- mediapipe 0.10.31 (latest compatible version)
- protobuf 6.33.3
- numpy 2.2.6
- opencv-python 4.12.0.88

**`mp_env` (Worker Environment - Python 3.10.19)**
- mediapipe 0.10.14 (legacy version with Holistic support)
- protobuf 4.25.8
- numpy 2.2.6
- opencv-python 4.12.0.88

### Architecture Pattern

The main app runs in `.venv` and spawns the MediaPipe worker in `mp_env` as a subprocess. They communicate via stdin/stdout using a binary protocol. This allows:

1. TensorFlow 2.20+ to run with its required protobuf 6.x
2. MediaPipe Holistic (0.10.14) to run with its required protobuf 4.x
3. Clean separation of concerns and dependency isolation

## Environment Setup

### Main Environment (.venv)

```bash
# Create environment (uses Python from pyproject.toml: 3.11+)
uv sync

# Verify installation
uv run python -c "import tensorflow, mediapipe; print(f'TF: {tensorflow.__version__}, MP: {mediapipe.__version__}')"
```

### MediaPipe Worker Environment (mp_env)

```bash
# Create Python 3.10 environment
uv venv mp_env --python 3.10

# Install legacy MediaPipe and dependencies
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy

# Verify installation
mp_env/bin/python -c "import mediapipe; print(f'MP: {mediapipe.__version__}')"
```

### Verify Both Environments

```bash
# Check main environment
uv run python -c "import tensorflow; print(f'protobuf: {tensorflow.__version__}')"
# Expected: protobuf: 6.33.3

# Check worker environment
mp_env/bin/python -c "import protobuf; print(f'protobuf: {protobuf.__version__}')"
# Expected: protobuf: 4.25.8
```

## MediaPipe Worker Protocol

### Communication Channel

The main app and MediaPipe worker communicate via stdin/stdout using a binary protocol.

### Input Format (Main → Worker)

```
Header:  8 bytes
  - height: 4 bytes (int32, little-endian)
  - width:  4 bytes (int32, little-endian)

Frame:   N bytes
  - N = height × width × 3 (BGR data, uint8)
```

### Output Format (Worker → Main)

```
Landmarks: 6,516 bytes
  - 543 landmarks × 3 coordinates (x, y, z) × 4 bytes/float32
  - Array shape: (543, 3)
  - dtype: float32
```

### Landmark Layout (Kaggle GISLR format)

- **Face**: indices 0-467 (468 points)
- **Left hand**: indices 468-488 (21 points)
- **Pose**: indices 489-521 (33 points)
- **Right hand**: indices 522-542 (21 points)

### Implementation

See `MediaPipeWorker` class in detect_signs.py:58-122 for the main app side, and `main()` in mp_worker.py:56-109 for the worker side.

## Troubleshooting

### Worker Startup Failures

**Symptom:** RuntimeError or FileNotFoundError when starting detect_signs.py

**Solutions:**
```bash
# Check if mp_env exists
ls -la mp_env/bin/python

# Check MediaPipe installation in mp_env
mp_env/bin/python -c "import mediapipe"

# Reinstall if needed
uv pip uninstall --python mp_env/bin/python mediapipe
uv pip install --python mp_env/bin/python mediapipe==0.10.14
```

### Incomplete Landmarks

**Symptom:** "Worker returned incomplete landmarks" error

**Causes:**
- Worker process crashed
- Frame size mismatch
- Buffer overflow

**Debug:**
```bash
# Check if worker is running
ps aux | grep mp_worker.py

# Check for errors in worker stderr (see detect_signs.py:86)
# Worker errors are printed to stderr
```

### Process Hangs

**Symptom:** Application freezes during landmark extraction

**Solutions:**
```bash
# Kill stuck worker
pkill -f mp_worker.py

# Restart application
uv run detect_signs.py
```

### Performance Issues

**Symptom:** High CPU usage, slow frame rate

**Optimizations:**
- Reduce camera resolution in capture loop
- Skip frames between predictions
- Use smaller input size for model
- Consider GPU acceleration for TensorFlow

## Performance Considerations

### Subprocess Overhead

- **Communication latency:** ~1ms per frame (stdin/stdout)
- **Serialization overhead:** ~0.5ms per frame (numpy.tobytes)
- **Worker initialization:** 1 second delay (detect_signs.py:84)

### Data Flow

```
Webcam → detect_signs.py → subprocess → mp_worker.py → MediaPipe
  ↓                                          ↓
 landmarks (6,516 bytes)                    landmarks (543×3)
```

### Bottlenecks

1. **MediaPipe Holistic:** ~30-50ms per frame (most expensive)
2. **Data transfer:** ~1.5ms per frame (negligible)
3. **TensorFlow inference:** ~5-10ms per prediction (every 3 seconds)

### Optimization Tips

- Skip frames between predictions (already done: prediction every 3 seconds)
- Use lower camera resolution (640×480 instead of 1920×1080)
- Process only every N-th frame for landmark visualization
- Consider threading/asyncio for parallel processing

## Project-Specific Notes

### ASL Recognition Model
Recognizes isolated ASL signs from MediaPipe landmarks (face, pose, left_hand, right_hand). Predictions every 3 seconds.

### Key Files
- `model.tflite` - TensorFlow Lite model
- `train.csv` - Training data with sign labels
- `10042041.parquet` - Reference landmark structure
- `out.parquet` - Temporary captured landmarks
- `detect_signs.py` - ASL sign detection with MediaPipe worker subprocess
- `mp_worker.py` - MediaPipe Holistic landmark extraction worker (runs in mp_env)

### Sign Mappings
Derived from `train.csv['sign_ord']`: `SIGN2ORD` maps names to codes, `ORD2SIGN` maps codes to names.

### Gemini-Pro Integration
Constructs sentences from recognized ASL words. Always ignores "TV". Prompt engineering is critical—maintain existing prompt structure.

### Processing Pipeline
1. Capture webcam frame from main process
2. Send frame to MediaPipe worker subprocess via stdin/stdout
3. Worker runs MediaPipe Holistic detection
4. Worker returns landmarks (543×3 array) to main process
5. Accumulate 3 seconds → prediction via TFLite
6. Append unique signs to list
7. Display with OpenCV

### UI/UX
Font scale 2.0-2.5 for accessibility. Escape toggles sentence display. 'q' quits. Handle empty landmark results gracefully.

### Performance
Reset `all_landmarks` after each prediction. Track time with `time.time()`. Use `cv2.INTER_LINEAR` for image scaling.
