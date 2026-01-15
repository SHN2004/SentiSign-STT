"""
ASL Sign Detection + Sentence Generation (Ollama) + Emotion Overlay

Extends detect_signs_sentences.py by adding real-time facial emotion recognition
using the SentiSign emotion model (PyTorch) from temprepo/SentiSign.

Notes:
- Sentence generation is intentionally unchanged by emotion (display-only).
- Emotion detection uses OpenCV Haar face detection + a small CNN classifier.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import detect_signs as ds
from detect_signs_sentences import (  # noqa: F401
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_WARMUP_TIMEOUT_SECONDS,
    PROMPT_TEMPLATE,
    _extract_three_sentences,
    _format_ollama_output,
    _ollama_generate,
    _wrap_index,
    _wrap_text,
)


def _add_temprepo_sentisign_to_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    sentisign_root = repo_root / "temprepo" / "SentiSign"
    if not sentisign_root.exists():
        raise FileNotFoundError(
            f"Expected SentiSign emotion repo at {sentisign_root}. "
            "Clone/copy it there so we can import sentisign.*."
        )
    sys.path.insert(0, str(sentisign_root))
    return sentisign_root


class EmotionRecognizer:
    EMO_ENTER_CONFIDENCE = 0.55
    EMO_EXIT_CONFIDENCE = 0.40
    EMO_STABILITY_COUNT = 2
    EMO_EXIT_COUNT = 1
    EMO_CLEAR_AFTER_SECONDS = 0.8
    BOX_EMA_ALPHA = 0.55
    SNAP_IOU_THRESHOLD = 0.15
    MIN_IOU_FOR_MATCH = 0.05

    def __init__(
        self,
        *,
        sentisign_root: Path,
        model_path: Path,
        device_pref: str,
        img_size: int,
        scale_factor: float,
        min_neighbors: int,
        clahe: bool,
    ) -> None:
        try:
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "PyTorch is required for emotion detection. Add/install 'torch' in the main environment."
            ) from e

        from sentisign.emotion.inference import EMOTION_LABELS, load_model, predict_emotions_batch
        from sentisign.utils.device import get_device, print_device_info
        from sentisign.utils.video import create_clahe, detect_faces, load_face_detector

        self._emotion_labels = EMOTION_LABELS
        self._predict_emotions_batch = predict_emotions_batch
        self._detect_faces = detect_faces
        self._get_device = get_device

        if not model_path.is_absolute():
            model_path = (sentisign_root / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Emotion model not found at {model_path}")

        self.device = self._get_device(device_pref)
        print_device_info(self.device)

        print(f"Loading emotion model from {model_path}...")
        self.model = load_model(
            model_path=str(model_path),
            num_classes=len(self._emotion_labels),
            in_channels=3,
            linear_in_features=2048,
            device=self.device,
        )
        print("Emotion model loaded successfully!")

        self.face_detector = load_face_detector()
        self.img_size = int(img_size)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.clahe = create_clahe() if clahe else None
        # This combined script intentionally limits emotion detection to a single face.

        self._locked_emotion: str | None = None
        self._locked_below_count = 0
        self._candidate_emotion: str | None = None
        self._candidate_count = 0
        self._last_face_time = 0.0
        self._last_emotion_label = "No face"
        self._last_emotion_conf = 0.0
        self._smoothed_face_box: tuple[float, float, float, float] | None = None

    @staticmethod
    def _box_iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0

        union = float((aw * ah) + (bw * bh) - (iw * ih))
        if union <= 0.0:
            return 0.0
        return inter / union

    @staticmethod
    def _ema_box(
        prev: tuple[float, float, float, float],
        new: tuple[int, int, int, int],
        alpha: float,
    ) -> tuple[float, float, float, float]:
        px, py, pw, ph = prev
        nx, ny, nw, nh = new
        return (
            (alpha * px) + ((1.0 - alpha) * float(nx)),
            (alpha * py) + ((1.0 - alpha) * float(ny)),
            (alpha * pw) + ((1.0 - alpha) * float(nw)),
            (alpha * ph) + ((1.0 - alpha) * float(nh)),
        )

    @staticmethod
    def _box_float_to_int(
        box: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        x, y, bw, bh = box
        return int(round(x)), int(round(y)), int(round(bw)), int(round(bh))

    @staticmethod
    def _clip_box_to_frame(
        box: tuple[float, float, float, float],
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x, y, bw, bh = box
        x = int(round(x))
        y = int(round(y))
        bw = int(round(bw))
        bh = int(round(bh))

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        return x, y, bw, bh

    def _pick_faces(self, faces: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not faces:
            return []
        if self._smoothed_face_box is None:
            largest = max(faces, key=lambda b: int(b[2]) * int(b[3]))
            return [largest]

        prev_int = self._box_float_to_int(self._smoothed_face_box)
        best = max(faces, key=lambda b: self._box_iou(prev_int, b))
        best_iou = self._box_iou(prev_int, best)
        if best_iou < self.MIN_IOU_FOR_MATCH:
            best = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        return [best]

    def _update_lock(self, pred_label: str, pred_conf: float) -> tuple[str, float, bool]:
        if self._locked_emotion is None:
            if self._candidate_emotion == pred_label:
                self._candidate_count += 1
            else:
                self._candidate_emotion = pred_label
                self._candidate_count = 1

            if self._candidate_count >= self.EMO_STABILITY_COUNT and pred_conf >= self.EMO_ENTER_CONFIDENCE:
                self._locked_emotion = pred_label
                self._locked_below_count = 0
        else:
            if pred_label == self._locked_emotion:
                if pred_conf < self.EMO_EXIT_CONFIDENCE:
                    self._locked_below_count += 1
                else:
                    self._locked_below_count = 0

                if self._locked_below_count >= self.EMO_EXIT_COUNT:
                    self._locked_emotion = None
                    self._candidate_emotion = None
                    self._candidate_count = 0
                    self._locked_below_count = 0
            else:
                if self._candidate_emotion == pred_label:
                    self._candidate_count += 1
                else:
                    self._candidate_emotion = pred_label
                    self._candidate_count = 1

                if self._candidate_count >= self.EMO_STABILITY_COUNT and pred_conf >= self.EMO_ENTER_CONFIDENCE:
                    self._locked_emotion = pred_label
                    self._locked_below_count = 0
                    self._candidate_emotion = None
                    self._candidate_count = 0

        label = self._locked_emotion if self._locked_emotion is not None else pred_label
        locked = self._locked_emotion is not None
        return label, pred_conf, locked

    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        now: float,
        detect: bool = True,
    ) -> tuple[list[tuple[tuple[int, int, int, int], str, float]], str, float, bool]:
        if not detect:
            if now - self._last_face_time >= self.EMO_CLEAR_AFTER_SECONDS:
                self._locked_emotion = None
                self._candidate_emotion = None
                self._candidate_count = 0
                self._locked_below_count = 0
                self._last_emotion_label = "No face"
                self._last_emotion_conf = 0.0
                self._smoothed_face_box = None
                return [], self._last_emotion_label, self._last_emotion_conf, False

            if self._smoothed_face_box is None:
                return [], self._last_emotion_label, self._last_emotion_conf, (self._locked_emotion is not None)

            box_int = self._clip_box_to_frame(self._smoothed_face_box, frame_bgr.shape)
            annotations = [(box_int, self._last_emotion_label, self._last_emotion_conf)]
            return annotations, self._last_emotion_label, self._last_emotion_conf, (self._locked_emotion is not None)

        faces = self._detect_faces(
            frame_bgr,
            self.face_detector,
            scale_factor=self.scale_factor,
            min_neighbors=self.min_neighbors,
        )
        faces = self._pick_faces(faces)

        if not faces:
            if now - self._last_face_time >= self.EMO_CLEAR_AFTER_SECONDS:
                self._locked_emotion = None
                self._candidate_emotion = None
                self._candidate_count = 0
                self._locked_below_count = 0
                self._last_emotion_label = "No face"
                self._last_emotion_conf = 0.0
                self._smoothed_face_box = None
            return [], self._last_emotion_label, self._last_emotion_conf, False

        self._last_face_time = now

        annotations = self._predict_emotions_batch(
            self.model,
            frame_bgr,
            faces,
            self.device,
            img_size=self.img_size,
            in_channels=3,
            imagenet_norm=True,
            clahe=self.clahe,
        )

        if not annotations:
            return [], self._last_emotion_label, self._last_emotion_conf, False

        (box, pred_label, pred_conf) = annotations[0]
        if self._smoothed_face_box is None:
            self._smoothed_face_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        else:
            prev_int = self._box_float_to_int(self._smoothed_face_box)
            iou = self._box_iou(prev_int, box)
            if iou < self.SNAP_IOU_THRESHOLD:
                # If the face moved significantly (or detector jittered), snap quickly.
                self._smoothed_face_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            else:
                self._smoothed_face_box = self._ema_box(self._smoothed_face_box, box, self.BOX_EMA_ALPHA)
        smooth_box_int = self._clip_box_to_frame(self._smoothed_face_box, frame_bgr.shape)

        stable_label, stable_conf, locked = self._update_lock(pred_label, float(pred_conf))

        self._last_emotion_label = stable_label
        self._last_emotion_conf = float(stable_conf)

        annotations = [(smooth_box_int, stable_label, float(stable_conf))]

        return annotations, stable_label, float(stable_conf), locked


class SentenceSignEmotionDetector(ds.SignDetector):
    def __init__(
        self,
        *,
        ollama_model: str,
        enable_ollama: bool,
        emotion: EmotionRecognizer,
    ) -> None:
        super().__init__()
        self.ollama_model = ollama_model
        self.enable_ollama = enable_ollama
        self.emotion = emotion
        self._ollama_ready = not enable_ollama

        self.words: list[str] = []
        self._last_word: str | None = None
        self._last_word_time = 0.0
        self._last_word_time_by_word: dict[str, float] = {}

        self._gen_lock = threading.Lock()
        self._gen_thread: threading.Thread | None = None
        self._gen_status: str = "Idle"
        self._gen_output: str = ""
        self._gen_error: str = ""
        self._sentences: list[str] = ["", "", ""]
        self._selected_sentence_idx = 0

        if self.enable_ollama:
            self._start_warmup()

    def _start_warmup(self) -> None:
        with self._gen_lock:
            if self._gen_thread is not None and self._gen_thread.is_alive():
                return
            self._gen_status = "Warming up..."
            self._gen_error = ""
            self._gen_output = ""

            def _worker():
                try:
                    _ollama_generate(
                        model=self.ollama_model,
                        prompt="Reply with exactly: READY",
                        timeout_s=OLLAMA_WARMUP_TIMEOUT_SECONDS,
                    )
                    self._ollama_ready = True
                    self._gen_status = "Ready"
                except Exception as e:  # noqa: BLE001
                    self._ollama_ready = False
                    self._gen_error = str(e)
                    self._gen_status = "Warmup error"

            self._gen_thread = threading.Thread(target=_worker, daemon=True)
            self._gen_thread.start()

    def _maybe_add_word(self, word: str, now: float) -> None:
        from detect_signs_sentences import (  # local import keeps this file minimal
            MAX_WORDS,
            MIN_SECONDS_BETWEEN_SAME_WORD,
            MIN_SECONDS_BETWEEN_WORDS,
        )

        if not word or word == "Waiting...":
            return

        if len(self.words) >= MAX_WORDS:
            return

        if now - self._last_word_time < MIN_SECONDS_BETWEEN_WORDS:
            return

        last_time_for_word = self._last_word_time_by_word.get(word, 0.0)
        if now - last_time_for_word < MIN_SECONDS_BETWEEN_SAME_WORD:
            return

        if self._last_word == word:
            return

        self.words.append(word)
        self._last_word = word
        self._last_word_time = now
        self._last_word_time_by_word[word] = now

    def _start_generation(self) -> None:
        if not self.enable_ollama:
            self._gen_error = "Ollama disabled (run without --no-ollama)."
            return

        if not self._ollama_ready:
            self._gen_error = "Ollama is still warming up."
            return

        if not self.words:
            self._gen_error = "No words captured yet."
            return

        with self._gen_lock:
            if self._gen_thread is not None and self._gen_thread.is_alive():
                return
            self._gen_status = "Generating..."
            self._gen_error = ""
            self._gen_output = ""
            self._sentences = ["", "", ""]
            self._selected_sentence_idx = 0

            prompt = PROMPT_TEMPLATE.format(
                words=" ".join(self.words),
            )

            def _worker():
                try:
                    out = _ollama_generate(
                        model=self.ollama_model,
                        prompt=prompt,
                        timeout_s=OLLAMA_TIMEOUT_SECONDS,
                    )
                    formatted = _format_ollama_output(out, allowed_words=self.words)
                    sentences = _extract_three_sentences(formatted)
                    with self._gen_lock:
                        self._gen_output = formatted
                        self._sentences = sentences
                        self._selected_sentence_idx = 0
                        self._gen_status = "Done"
                except Exception as e:  # noqa: BLE001
                    with self._gen_lock:
                        self._gen_error = str(e)
                        self._gen_status = "Error"

            self._gen_thread = threading.Thread(target=_worker, daemon=True)
            self._gen_thread.start()

    def run(
        self,
        *,
        camera_index: int = 0,
        flip: bool = True,
        emotion_every_n_frames: int = 3,
    ) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            cap.release()
            return

        frame_height, frame_width = frame.shape[:2]
        display_width = frame_width + 620

        frame_buffer = deque(maxlen=ds.BUFFER_SIZE)
        frame_count = 0
        start_time = time.time()
        current_sign = "Waiting..."
        current_confidence = 0.0
        ema_proba = None
        candidate_sign = None
        candidate_count = 0
        locked_sign = None
        locked_sign_below_count = 0

        emotion_label = "No face"
        emotion_conf = 0.0
        emotion_locked = False

        font = cv2.FONT_HERSHEY_SIMPLEX

        print("\nSign + Emotion + Sentence Generation Started")
        print("Keys: q quit | a add word | s generate | c clear | b backspace | arrows/jk/1-3 select")
        print("-" * 70)

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                if flip:
                    image = cv2.flip(image, 1)

                now = time.time()
                elapsed = now - start_time

                landmarks = self.mp_worker.extract_landmarks(image)
                frame_buffer.append(landmarks)
                frame_count += 1

                should_infer = (
                    frame_count % ds.INFER_EVERY_N_FRAMES == 0
                    and len(frame_buffer) >= ds.MIN_FRAMES_FOR_INFERENCE
                )

                if should_infer:
                    frames, frame_idxs = self.preprocess(list(frame_buffer))
                    if frames is not None:
                        proba = self.predict_proba(frames, frame_idxs)
                        if ema_proba is None:
                            ema_proba = proba.astype(np.float32, copy=True)
                        else:
                            ema_proba = (ds.EMA_ALPHA * ema_proba) + ((1.0 - ds.EMA_ALPHA) * proba)

                        pred_idx = int(np.argmax(ema_proba))
                        pred_sign = self.ord2sign[pred_idx]
                        pred_conf = float(ema_proba[pred_idx])

                        if locked_sign is None:
                            if candidate_sign == pred_sign:
                                candidate_count += 1
                            else:
                                candidate_sign = pred_sign
                                candidate_count = 1

                            if candidate_count >= ds.STABILITY_COUNT and pred_conf >= ds.ENTER_CONFIDENCE:
                                locked_sign = pred_sign
                                locked_sign_below_count = 0
                                self._maybe_add_word(locked_sign, now)
                        else:
                            locked_idx = self.sign2ord.get(locked_sign)
                            if locked_idx is not None:
                                locked_conf = float(ema_proba[int(locked_idx)])
                                if locked_conf < ds.EXIT_CONFIDENCE:
                                    locked_sign_below_count += 1
                                else:
                                    locked_sign_below_count = 0

                                if locked_sign_below_count >= ds.STABILITY_COUNT:
                                    locked_sign = None
                                    candidate_sign = None
                                    candidate_count = 0
                                    locked_sign_below_count = 0

                        if locked_sign is not None:
                            current_sign = locked_sign
                            locked_idx = self.sign2ord.get(locked_sign)
                            if locked_idx is not None:
                                current_confidence = float(ema_proba[int(locked_idx)])
                            else:
                                current_confidence = pred_conf
                        else:
                            current_sign = pred_sign
                            current_confidence = pred_conf

                should_emotion_infer = emotion_every_n_frames > 0 and (frame_count % emotion_every_n_frames == 0)
                (
                    emotion_annotations,
                    emotion_label,
                    emotion_conf,
                    emotion_locked,
                ) = self.emotion.infer(image, now=now, detect=should_emotion_infer)

                display_image = self.draw_landmarks(image.copy(), landmarks)
                if emotion_annotations:
                    from sentisign.utils.video import draw_annotations

                    draw_annotations(display_image, emotion_annotations, color=(0, 255, 0))

                display = np.zeros((frame_height, display_width, 3), dtype=np.uint8)
                display[:, :frame_width] = display_image

                panel_x = frame_width + 20
                line_h = 20

                cv2.putText(display, "ASL + Emotion", (panel_x, 40),
                            font, 0.8, (255, 255, 255), 2)
                cv2.putText(display, "-" * 22, (panel_x, 70),
                            font, 0.5, (100, 100, 100), 1)

                cv2.putText(display, "Detected Sign:", (panel_x, 110),
                            font, 0.6, (200, 200, 200), 1)
                cv2.putText(display, current_sign, (panel_x, 145),
                            font, 1.1, (0, 255, 0), 2)
                cv2.putText(display, f"Confidence: {current_confidence:.1%}", (panel_x, 175),
                            font, 0.5, (200, 200, 200), 1)

                emo_lock_status = "LOCKED" if emotion_locked else "UNLOCKED"
                cv2.putText(display, "Emotion:", (panel_x, 215),
                            font, 0.6, (200, 200, 200), 1)
                cv2.putText(display, f"{emotion_label} ({emotion_conf:.0%})", (panel_x, 245),
                            font, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Emotion mode: rolling ({emo_lock_status})", (panel_x, 270),
                            font, 0.45, (150, 150, 150), 1)

                lock_status = "LOCKED" if locked_sign is not None else "UNLOCKED"
                cv2.putText(display, f"Sign mode: rolling ({lock_status})", (panel_x, 295),
                            font, 0.45, (150, 150, 150), 1)
                cv2.putText(display, f"Elapsed: {int(elapsed)}s", (panel_x, 315),
                            font, 0.45, (150, 150, 150), 1)
                cv2.putText(display, f"Buffer: {len(frame_buffer)}/{ds.BUFFER_SIZE}", (panel_x, 335),
                            font, 0.45, (150, 150, 150), 1)

                y = 370
                cv2.putText(display, "Words:", (panel_x, y),
                            font, 0.55, (255, 255, 255), 1)
                y += line_h

                words_preview = " ".join(self.words[-8:]) if self.words else ""
                for wl in _wrap_text(words_preview, width=32)[:2]:
                    cv2.putText(display, wl, (panel_x, y),
                                font, 0.45, (220, 220, 220), 1)
                    y += line_h

                y += 8
                status = self._gen_status
                cv2.putText(display, f"Ollama: {status}", (panel_x, y),
                            font, 0.5, (200, 200, 200), 1)
                y += line_h

                if self._gen_error:
                    for wl in _wrap_text(f"Error: {self._gen_error}", width=32)[:4]:
                        cv2.putText(display, wl, (panel_x, y),
                                    font, 0.45, (0, 0, 255), 1)
                        y += line_h
                elif self._gen_output:
                    cv2.putText(display, "Sentences:", (panel_x, y),
                                font, 0.55, (255, 255, 255), 1)
                    y += line_h

                    selected = _wrap_index(self._selected_sentence_idx, 3)
                    for idx, sentence in enumerate(self._sentences[:3]):
                        prefix = f"{idx + 1}. "
                        is_selected = idx == selected
                        color = (0, 255, 255) if is_selected else (220, 220, 220)
                        label = ("> " if is_selected else "  ") + prefix + (sentence or "")
                        for wl in _wrap_text(label, width=32)[:2]:
                            cv2.putText(display, wl, (panel_x, y),
                                        font, 0.45, color, 1)
                            y += line_h

                cv2.putText(display, "q quit | a add word | s generate | c clear | b backspace",
                            (panel_x, frame_height - 30),
                            font, 0.45, (120, 120, 120), 1)
                cv2.putText(display, "arrows/jk/1-3 select", (panel_x, frame_height - 10),
                            font, 0.45, (120, 120, 120), 1)

                cv2.imshow("ASL Sign Detection (Emotion + Sentences)", display)

                key = cv2.waitKeyEx(5)
                if key == -1:
                    continue

                key_char = key & 0xFF
                if key_char == ord("q"):
                    break
                if key_char == ord("a"):
                    if locked_sign is not None:
                        self._maybe_add_word(locked_sign, now)
                if key_char == ord("c"):
                    self.words.clear()
                    self._last_word = None
                    self._last_word_time = 0.0
                    self._last_word_time_by_word.clear()
                    self._gen_output = ""
                    self._gen_error = ""
                    self._gen_status = "Idle"
                    self._sentences = ["", "", ""]
                    self._selected_sentence_idx = 0
                if key_char == ord("b"):
                    if self.words:
                        removed = self.words.pop()
                        if removed == self._last_word:
                            self._last_word = self.words[-1] if self.words else None
                    self._gen_output = ""
                    self._gen_error = ""
                    self._gen_status = "Idle"
                    self._sentences = ["", "", ""]
                    self._selected_sentence_idx = 0

                up_keys = {2490368, 63232, 65362}
                down_keys = {2621440, 63233, 65364}
                left_keys = {2424832, 63234, 65361}
                right_keys = {2555904, 63235, 65363}

                if key in up_keys or key in left_keys or key_char in {ord("k")}:
                    if any(self._sentences):
                        self._selected_sentence_idx = _wrap_index(self._selected_sentence_idx - 1, 3)
                if key in down_keys or key in right_keys or key_char in {ord("j")}:
                    if any(self._sentences):
                        self._selected_sentence_idx = _wrap_index(self._selected_sentence_idx + 1, 3)
                if key_char in {ord("1"), ord("2"), ord("3")}:
                    if any(self._sentences):
                        self._selected_sentence_idx = int(chr(key_char)) - 1

                if key_char == ord("s"):
                    self._start_generation()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
            print("\nStopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="ASL detection + emotion overlay + Ollama sentences")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")

    parser.add_argument("--flip", action="store_true", default=True, help="Flip frame horizontally (default)")
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="Disable flip")

    parser.add_argument("--clahe", action="store_true", default=True, help="Use CLAHE for emotion preprocessing (default)")
    parser.add_argument("--no-clahe", dest="clahe", action="store_false", help="Disable CLAHE")

    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Emotion compute device")
    parser.add_argument(
        "--emotion-model",
        type=Path,
        default=Path("models/emotion/resnet_emotion.pth"),
        help="Path to emotion model (relative to temprepo/SentiSign unless absolute)",
    )
    parser.add_argument("--emotion-every-n-frames", type=int, default=3, help="Run emotion inference every N frames (default: 3)")
    parser.add_argument("--scale-factor", type=float, default=1.3, help="Face detection scale factor")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Face detection minimum neighbors")
    parser.add_argument("--img-size", type=int, default=44, help="Emotion model input size (default: 44)")

    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--no-ollama", action="store_true", help="Disable Ollama generation (still buffers words)")
    args = parser.parse_args()

    sentisign_root = _add_temprepo_sentisign_to_path()
    emotion = EmotionRecognizer(
        sentisign_root=sentisign_root,
        model_path=args.emotion_model,
        device_pref=args.device,
        img_size=args.img_size,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        clahe=bool(args.clahe),
    )

    detector = SentenceSignEmotionDetector(
        ollama_model=args.ollama_model,
        enable_ollama=not args.no_ollama,
        emotion=emotion,
    )
    detector.run(
        camera_index=args.camera,
        flip=bool(args.flip),
        emotion_every_n_frames=int(args.emotion_every_n_frames),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
