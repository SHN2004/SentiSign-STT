# SignSense (SentiSign-STT)

## Project Overview
SignSense is a real-time American Sign Language (ASL) recognition system. It leverages computer vision and deep learning to translate ASL gestures into text. The system captures live video, extracts holistic landmarks (face, pose, hands), processes them through a custom Transformer-based TensorFlow Lite model, and uses the Gemini-Pro LLM to construct coherent English sentences from the recognized signs.

## Architecture & Technology
*   **Language:** Python
*   **Computer Vision:** OpenCV, MediaPipe (Holistic)
*   **ML Framework:** TensorFlow / TensorFlow Lite, Keras
*   **LLM Integration:** Google Gemini-Pro API (via `google-generativeai`)
*   **Data Processing:** Pandas, NumPy, Parquet

## Build & Run

### Prerequisites
*   Python 3.x
*   A valid Google Gemini API Key

### Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up environment variables:
    *   Create a `.env` file in the root directory.
    *   Add your API key: `GOOGLE_API_KEY=your_api_key_here`

### Execution
*   **Run the Application:**
    ```bash
    python app.py
    ```
    *   **Controls:**
        *   `Esc`: Toggle sentence generation (sends recognized signs to Gemini).
        *   `q`: Quit the application.

*   **Train the Model:**
    Open and run the Jupyter notebook:
    ```bash
    jupyter notebook ASL_Model_Training.ipynb
    ```

## Key Files & Directories
*   `app.py`: The main application entry point. Handles video capture, landmark extraction, inference loop, and UI rendering.
*   `requirements.txt`: Python package dependencies.
*   `ASL_Model_Training.ipynb`: Jupyter notebook used for training the ASL recognition model.
*   `model.tflite`: The pre-trained TensorFlow Lite model used for real-time inference.
*   `train.csv`: Contains the mapping between sign names and their ordinal labels (`SIGN2ORD` / `ORD2SIGN`).
*   `10042041.parquet`: A sample/reference Parquet file used to establish the dataframe schema for landmarks.

## Development Conventions
*   **Environment:** Uses `.env` for secrets management.
*   **Data Flow:**
    1.  **Capture:** OpenCV reads frames.
    2.  **Process:** MediaPipe extracts 543 landmarks per frame.
    3.  **Buffer:** Landmarks are accumulated (approx. 3 seconds).
    4.  **Inference:** TFLite model predicts the sign.
    5.  **Refine:** Gemini API constructs a sentence from the list of predicted signs.
*   **UI:** Uses `cv2.imshow` for a simple real-time display with overlaid text.
