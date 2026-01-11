# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SignSense is a real-time American Sign Language (ASL) recognition system using a transformer-based deep learning model. It captures video from a webcam, extracts hand/face/pose landmarks via MediaPipe, runs inference with a TensorFlow Lite model, and uses the Gemini-Pro API to construct sentences from recognized signs.

## Commands

### Run the Application
```bash
python app.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
jupyter notebook ASL_Model_Training.ipynb
```

## Architecture

### Data Flow Pipeline
1. **Video Capture**: OpenCV captures frames from webcam (`cv2.VideoCapture(0)`)
2. **Landmark Extraction**: MediaPipe Holistic extracts 543 landmarks per frame (face, pose, left_hand, right_hand)
3. **Data Accumulation**: Landmarks accumulated for 3 seconds, stored as parquet
4. **Inference**: TensorFlow Lite model predicts ASL sign from landmark sequences
5. **Sentence Construction**: Gemini-Pro API reorders recognized words into coherent sentences

### Key Data Structures
- **Landmark DataFrame**: 543 rows per frame with columns `[type, landmark_index, x, y, z, frame]`
- **Sign Mappings**: `SIGN2ORD` (sign name → ordinal) and `ORD2SIGN` (ordinal → sign name) derived from `train.csv`

### Model Architecture (in notebook)
- Custom transformer with multi-head attention
- Input: 64 frames × 66 landmarks × 3 dimensions (preprocessed from 543 landmarks)
- Separate embeddings for lips (40), left_hand (21), and pose (5) landmarks
- 2 transformer blocks with 512-dimensional embeddings
- Output: 250 ASL sign classes

### Required Files
- `model.tflite` - Trained TensorFlow Lite model for inference
- `train.csv` - Training metadata with sign labels (creates sign dictionaries)
- `10042041.parquet` - Reference landmark structure for DataFrame schema
- `.env` - Must contain `GOOGLE_API_KEY` for Gemini-Pro API

## Environment Variables

```bash
GOOGLE_API_KEY=<your-gemini-api-key>
```

## UI Controls
- **Escape**: Toggle sentence display (triggers Gemini API call)
- **q**: Quit application
