# SignSense (SentiSign-STT)

## Project Overview
SignSense is a real-time American Sign Language (ASL) recognition system. It leverages computer vision and deep learning to translate ASL gestures into text. The system captures live video, extracts holistic landmarks (face, pose, hands), processes them through a custom Transformer-based TensorFlow Lite model, and uses the Gemini-Pro LLM to construct coherent English sentences from the recognized signs.

Note: Gemini sentence generation is currently only wired up in the legacy pipeline (`app.py` / some notebook code). The recommended runtime (`detect_signs.py`) does not use Gemini.

## Architecture & Technology
*   **Language:** Python
*   **Computer Vision:** OpenCV, MediaPipe (Holistic)
*   **ML Framework:** TensorFlow / TensorFlow Lite, Keras
*   **LLM Integration (legacy):** Google Gemini-Pro API (via `google-generativeai`)
*   **Data Processing:** NumPy (recommended pipeline); Pandas/Parquet (legacy pipeline)

## Build & Run

### Prerequisites
*   Python 3.11+ for the main app (`uv sync`)
*   Python 3.10 for the MediaPipe worker (`mp_env`)
*   A valid Google Gemini API key only if you want to use the legacy Gemini flow

### Installation
1.  Install dependencies:
    ```bash
    uv sync
    ```
2.  Set up environment variables:
    *   Create a `.env` file in the root directory.
    *   Add your API key: `GOOGLE_API_KEY=your_api_key_here`

### Execution
*   **Run Sign Detection (recommended):**
    ```bash
    uv run detect_signs.py
    ```
    *   **Controls:**
        *   `q`: Quit the application.

*   **Run the Legacy Gemini Application (not guaranteed):**
    ```bash
    uv run app.py
    ```
    *   **Controls:**
        *   `Esc`: Toggle sentence generation (sends recognized signs to Gemini).
        *   `q`: Quit the application.

*   **Train the Model:**
    Open and run the Jupyter notebook:
    ```bash
    uv run jupyter notebook ASL_Model_Training.ipynb
    ```

## Key Files & Directories
*   `detect_signs.py`: Recommended application entry point (worker-based landmark extraction + TFLite inference).
*   `mp_worker.py`: MediaPipe Holistic worker subprocess (runs under `mp_env`).
*   `app.py`: Legacy application entry point (Gemini + Parquet-based pipeline).
*   `pyproject.toml` / `uv.lock`: Python dependencies (uv-managed).
*   `ASL_Model_Training.ipynb`: Jupyter notebook used for training the ASL recognition model.
*   `model_66landmarks.tflite`: The pre-trained TensorFlow Lite model used by `detect_signs.py`.
*   `label_mappings.json`: Ordinal label ↔ sign name mappings used by `detect_signs.py`.
*   `10042041.parquet`: Reference Parquet file used in older notebook/legacy workflows.

## Development Conventions
*   **Environment:** Uses `.env` for secrets management.
*   **Data Flow:**
    1.  **Capture:** OpenCV reads frames.
    2.  **Process:** `detect_signs.py` sends frames to the MediaPipe worker for 543 landmarks per frame.
    3.  **Preprocess:** Reduce to 66 landmarks and normalize into the model’s input format.
    4.  **Inference:** TFLite model predicts the sign.
    5.  **Refine (legacy only):** Gemini API constructs a sentence from the list of predicted signs.
*   **UI:** Uses `cv2.imshow` for a simple real-time display with overlaid text.
