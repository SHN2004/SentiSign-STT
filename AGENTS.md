# AGENTS.md

This file provides guidelines for agentic coding assistants working in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main application
python app.py

# Train model (run cells in Jupyter notebook)
jupyter notebook ASL_Model_Training.ipynb
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

## Project-Specific Notes

### ASL Recognition Model
Recognizes isolated ASL signs from MediaPipe landmarks (face, pose, left_hand, right_hand). Predictions every 3 seconds.

### Key Files
- `model.tflite` - TensorFlow Lite model
- `train.csv` - Training data with sign labels
- `10042041.parquet` - Reference landmark structure
- `out.parquet` - Temporary captured landmarks

### Sign Mappings
Derived from `train.csv['sign_ord']`: `SIGN2ORD` maps names to codes, `ORD2SIGN` maps codes to names.

### Gemini-Pro Integration
Constructs sentences from recognized ASL words. Always ignores "TV". Prompt engineering is critical—maintain existing prompt structure.

### Processing Pipeline
1. Capture webcam frame → BGR to RGB
2. MediaPipe Holistic detection
3. Extract landmarks (543 rows: face/pose/hands)
4. Accumulate 3 seconds → prediction via TFLite
5. Append unique signs to list
6. Display with OpenCV

### UI/UX
Font scale 2.0-2.5 for accessibility. Escape toggles sentence display. 'q' quits. Handle empty landmark results gracefully.

### Performance
Reset `all_landmarks` after each prediction. Track time with `time.time()`. Use `cv2.INTER_LINEAR` for image scaling.
