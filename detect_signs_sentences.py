"""
ASL Sign Detection + Sentence Generation (Ollama)

Runs the same detection pipeline as detect_signs.py, but buffers detected words and,
on keypress, sends them to a local Ollama model to generate sentence variations.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

import detect_signs as ds


DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.3:2b")
OLLAMA_TIMEOUT_SECONDS = 60
OLLAMA_WARMUP_TIMEOUT_SECONDS = 180

# Word capture tuning
MIN_SECONDS_BETWEEN_WORDS = 0.8
MIN_SECONDS_BETWEEN_SAME_WORD = 2.0
MAX_WORDS = 20

MAX_EXTRA_WORDS_PER_SENTENCE = 6

PROMPT_TEMPLATE = """You are helping convert ASL sign detections into natural English.

Input words (in order): {words}

Task:
- Produce exactly 3 different sentence variations.
- Use the input words as the core content words; you may reorder or omit some if needed.
- You MAY add a few extra words to make it grammatical, but keep additions minimal.
- Keep each sentence short (aim for <= 10 words) and avoid extra descriptive details.

Output format (exactly 3 lines):
1. <sentence>
2. <sentence>
3. <sentence>
"""


def _wrap_text(text: str, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        paragraph = paragraph.rstrip()
        if not paragraph:
            lines.append("")
            continue
        while len(paragraph) > width:
            cut = paragraph.rfind(" ", 0, width + 1)
            if cut == -1:
                cut = width
            lines.append(paragraph[:cut].rstrip())
            paragraph = paragraph[cut:].lstrip()
        lines.append(paragraph)
    return lines


def _ollama_generate(model: str, prompt: str, timeout_s: int) -> str:
    if shutil.which("ollama") is None:
        raise RuntimeError("ollama CLI not found in PATH. Install Ollama and try again.")

    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        msg = f"ollama exited with code {result.returncode}"
        if stderr:
            msg += f": {stderr}"
        raise RuntimeError(msg)

    return (result.stdout or "").strip()


def _strip_leading_numbering(line: str) -> str:
    line = line.strip()
    if len(line) >= 2 and line[0] in {"1", "2", "3"}:
        line = line[1:].lstrip()
        if line.startswith("."):
            line = line[1:].lstrip()
    return line


def _minimalize_sentence(sentence: str, *, allowed_words: list[str]) -> str:
    allowed_lower = {w.lower() for w in allowed_words if w.strip()}
    tokens: list[str] = []
    extra_used = 0

    raw_tokens = sentence.replace("\t", " ").split()
    for raw in raw_tokens:
        stripped = raw.strip(".,!?;:\"'()[]{}")
        if not stripped:
            continue

        lower = stripped.lower()
        if lower in allowed_lower:
            tokens.append(lower)
            continue

        if extra_used < MAX_EXTRA_WORDS_PER_SENTENCE:
            tokens.append(lower)
            extra_used += 1

    if not tokens:
        tokens = [w.lower() for w in allowed_words if w.strip()]

    if allowed_lower and not any(t in allowed_lower for t in tokens):
        tokens = [w.lower() for w in allowed_words if w.strip()]

    out = " ".join(tokens).strip()
    if out:
        out = out[0].upper() + out[1:]
    return out


def _format_ollama_output(raw: str, *, allowed_words: list[str]) -> str:
    lines = [l for l in (raw or "").splitlines() if l.strip()]
    base = " ".join(allowed_words)

    candidates: list[str] = []
    for l in lines:
        l = _strip_leading_numbering(l)
        if l:
            candidates.append(l)
        if len(candidates) == 3:
            break

    while len(candidates) < 3:
        candidates.append(base)

    cleaned = [_minimalize_sentence(c, allowed_words=allowed_words) for c in candidates[:3]]
    return "1. " + cleaned[0] + "\n2. " + cleaned[1] + "\n3. " + cleaned[2]


class SentenceSignDetector(ds.SignDetector):
    def __init__(self, *, ollama_model: str, enable_ollama: bool):
        super().__init__()
        self.ollama_model = ollama_model
        self.enable_ollama = enable_ollama
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
                    self._gen_output = _format_ollama_output(out, allowed_words=self.words)
                    self._gen_status = "Done"
                except Exception as e:  # noqa: BLE001
                    self._gen_error = str(e)
                    self._gen_status = "Error"

            self._gen_thread = threading.Thread(target=_worker, daemon=True)
            self._gen_thread.start()

    def run(self, *, camera_index: int = 0):
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
        display_width = frame_width + 520

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

        font = cv2.FONT_HERSHEY_SIMPLEX

        print("\nSign Detection + Sentence Generation Started")
        print("Keys: q quit | a add word | s generate | c clear | b backspace")
        print("-" * 60)

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

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

                display_image = self.draw_landmarks(image.copy(), landmarks)
                display = np.zeros((frame_height, display_width, 3), dtype=np.uint8)
                display[:, :frame_width] = display_image

                panel_x = frame_width + 20
                y = 40
                line_h = 26

                cv2.putText(display, "ASL -> Words -> Sentences (Ollama)", (panel_x, y),
                            font, 0.65, (255, 255, 255), 2)
                y += line_h

                cv2.putText(display, f"Detected: {current_sign} ({current_confidence:.1%})", (panel_x, y),
                            font, 0.55, (200, 200, 200), 1)
                y += line_h

                lock_status = "LOCKED" if locked_sign is not None else "UNLOCKED"
                cv2.putText(display, f"Mode: rolling ({lock_status})", (panel_x, y),
                            font, 0.5, (150, 150, 150), 1)
                y += line_h

                cv2.putText(display, f"Elapsed: {int(elapsed)}s", (panel_x, y),
                            font, 0.5, (150, 150, 150), 1)
                y += line_h

                cv2.putText(display, f"Words ({len(self.words)}/{MAX_WORDS}):", (panel_x, y),
                            font, 0.55, (255, 255, 255), 1)
                y += line_h

                words_line = " ".join(self.words) if self.words else "(none yet)"
                for wl in _wrap_text(words_line, width=32)[:4]:
                    cv2.putText(display, wl, (panel_x, y),
                                font, 0.55, (0, 255, 0), 1)
                    y += line_h

                y += 6
                cv2.putText(display, f"Ollama model: {self.ollama_model}", (panel_x, y),
                            font, 0.45, (150, 150, 150), 1)
                y += line_h

                status_line = self._gen_status
                if self.enable_ollama and not self._ollama_ready and self._gen_status == "Idle":
                    status_line = "Warming up..."
                cv2.putText(display, f"Status: {status_line}", (panel_x, y),
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
                    for wl in _wrap_text(self._gen_output, width=32)[:9]:
                        cv2.putText(display, wl, (panel_x, y),
                                    font, 0.45, (220, 220, 220), 1)
                        y += line_h

                cv2.putText(display, "q quit | a add word | s generate | c clear | b backspace",
                            (panel_x, frame_height - 30),
                            font, 0.45, (120, 120, 120), 1)

                cv2.imshow("ASL Sign Detection (Sentences)", display)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):
                    if locked_sign is not None:
                        self._maybe_add_word(locked_sign, now)
                if key == ord("c"):
                    self.words.clear()
                    self._last_word = None
                    self._last_word_time = 0.0
                    self._last_word_time_by_word.clear()
                    self._gen_output = ""
                    self._gen_error = ""
                    self._gen_status = "Idle"
                if key == ord("b"):
                    if self.words:
                        removed = self.words.pop()
                        if removed == self._last_word:
                            self._last_word = self.words[-1] if self.words else None
                    self._gen_output = ""
                    self._gen_error = ""
                    self._gen_status = "Idle"
                if key == ord("s"):
                    self._start_generation()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
            print("\nStopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="ASL detection + Ollama sentence generation")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--no-ollama", action="store_true", help="Disable Ollama generation (still buffers words)")
    args = parser.parse_args()

    detector = SentenceSignDetector(ollama_model=args.ollama_model, enable_ollama=not args.no_ollama)
    detector.run(camera_index=args.camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
