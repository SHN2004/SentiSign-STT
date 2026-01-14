# AGENTS.md

Guidelines for agentic coding assistants working in this repository.

## Source of truth

- Dependency management: `pyproject.toml` + `uv.lock` (not `requirements.txt`)
- Recommended runtime: `detect_signs.py` (worker-based)
- Legacy runtime: `app.py` (Gemini + Parquet; requires extra deps/files)

## Commands

```bash
# Main environment (Python 3.11+, from pyproject.toml)
uv sync

# MediaPipe worker environment (Python 3.10, legacy MediaPipe Holistic)
uv venv mp_env --python 3.10
uv pip install --python mp_env/bin/python mediapipe==0.10.14 opencv-python numpy

# Run (recommended)
uv run detect_signs.py

# Run (legacy; see “Legacy app.py”)
uv run app.py

# Train / explore
uv run jupyter notebook ASL_Model_Training.ipynb
```

**Testing**: no automated tests; validate manually by running the app(s).

## Project entrypoints

### `detect_signs.py` (recommended)

- Uses `mp_worker.py` via a subprocess (`mp_env/bin/python`)
- Runs inference with `model_66landmarks.tflite`
- Uses `label_mappings.json` for `ord -> sign` mapping
- Predictions every `PREDICTION_INTERVAL` seconds (default: 3)
- UI: press `q` to quit

### `mp_worker.py` (worker)

- Must run under `mp_env` with `mediapipe==0.10.14`
- Reads raw frames from stdin, writes landmarks to stdout (binary protocol below)

### `app.py` (legacy)

This pipeline is older and not guaranteed to run out-of-the-box with the current `pyproject.toml`.

- Requires additional dependencies not listed in `pyproject.toml` (notably `pandas`, `python-dotenv`, `google-generativeai`)
- Expects `model.tflite` and `train.csv` (not currently present in this repo)
- Uses `.env` (`GOOGLE_API_KEY`) for Gemini sentence generation

If you want `app.py` supported again, decide whether to:
- add its missing deps to `pyproject.toml`, and
- add/restore the expected model/data files (or update `app.py` to use `model_66landmarks.tflite` + `label_mappings.json`).

## Dual environment architecture

### Why two environments?

**Protobuf version conflict:**
- TensorFlow 2.20+ requires `protobuf >= 6.x`
- MediaPipe 0.10.14 requires `protobuf < 5.x`
- They cannot coexist in one environment reliably

### Environments

- **Main**: `.venv` (Python 3.11+, `uv sync`) for TensorFlow + the main UI loop
- **Worker**: `mp_env` (Python 3.10, manually created) for legacy MediaPipe Holistic extraction

### Verify installs

```bash
# Main env versions
uv run python -c "import tensorflow, mediapipe; print(f'TF: {tensorflow.__version__}, MP: {mediapipe.__version__}')"

# Protobuf versions
uv run python -c "import google.protobuf; print(f'protobuf: {google.protobuf.__version__}')"
mp_env/bin/python -c "import google.protobuf; print(f'protobuf: {google.protobuf.__version__}')"

# Worker env MediaPipe
mp_env/bin/python -c "import mediapipe; print(f'MP: {mediapipe.__version__}')"
```

## MediaPipe worker protocol

### Input (main → worker)

```
Header: 8 bytes total
  - height: int32
  - width:  int32

Frame:
  - height * width * 3 bytes (BGR, uint8)
```

Implementation note: code uses `struct.pack('<ii', height, width)` / `struct.unpack('<ii', ...)` (explicit little-endian int32).

### Output (worker → main)

```
Landmarks: 6516 bytes
  - 543 * 3 float32 values (shape: (543, 3))
  - NaN where a landmark is missing
```

### Landmark layout (Kaggle GISLR)

- Face: 0–467 (468)
- Left hand: 468–488 (21)
- Pose: 489–521 (33)
- Right hand: 522–542 (21)

## Model + preprocessing (detect_signs.py)

- Files: `model_66landmarks.tflite`, `label_mappings.json`
- TFLite inputs:
  - `frames`: `(1, 64, 66, 3)` `float32`
  - `frame_idxs`: `(1, 64)` `int32`
- Preprocessing:
  - select 66 landmarks: lips (40) + one hand (21) + pose (5)
  - choose dominant hand across frames; if right-dominant, use right-hand set and flip x
  - resize/pad to 64 frames; normalize relative to pose landmarks; compute `frame_idxs`

`summary.md` describes the expected input/output concisely.

## Coding conventions

### Imports

Standard library → third-party → local.

Some existing files don’t fully follow this; prefer the guideline for new/modified code when practical.

### Naming

- functions/vars: `snake_case`
- constants: `ALL_CAPS`
- classes: `PascalCase`
- mappings: use descriptive suffixes (`SIGN2ORD`, `ORD2SIGN`)

### Error handling

- Always validate webcam capture (`cap.isOpened()`, `ret`/`success`) before processing.
- Handle worker startup failures cleanly (missing `mp_env`, missing `mp_worker.py`, worker stderr).
- If reading/writing model or mapping files, raise/print a clear message with the expected path.

### Secrets

- Store secrets in `.env` (never commit it).
- Use `os.getenv("GOOGLE_API_KEY")` for Gemini (legacy flow).

## Troubleshooting

### Worker startup failures

```bash
ls -la mp_env/bin/python
mp_env/bin/python -c "import mediapipe; print(mediapipe.__version__)"
```

### Incomplete landmarks

If you see `Worker returned incomplete landmarks`, the worker likely crashed or stdout was interrupted:

```bash
ps aux | rg "mp_worker.py" || ps aux | grep mp_worker.py
```

### Process hangs

```bash
pkill -f mp_worker.py
uv run detect_signs.py
```

### Performance

- Reduce camera resolution (e.g., 640×480)
- Skip drawing every frame if needed
- Keep prediction interval conservative on slower machines
