import os
import gdown
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque

# =========================================================
# Load TFLite ISL Model
# =========================================================
FILE_ID = "1ma9c9NjI9wuFSF3KFFYACdjaKD2Tvh5w"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "isl_model.tflite"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================================================
# Load YOLO Hand Detector
# =========================================================
HAND_MODEL = "hand_yolov8n.pt"

if not os.path.exists(HAND_MODEL):
    st.error("❌ 'hand_yolov8n.pt' not found. Place it in the folder.")
    st.stop()

hand_detector = YOLO(HAND_MODEL)

# =========================================================
# Load TFLite Interpreter
# =========================================================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [line.strip().upper() for line in open("labels.txt")]

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="ISL Sign App", layout="wide")
st.title("ISL Sign Language App")

run = st.sidebar.checkbox("Start Webcam")

# Smoothing
pred_queue = deque(maxlen=20)
ema_scores = {}
ema_alpha = 0.25

locked_letter = None
lock_counter = 0
lock_frames = 25

frame_ph = st.empty()
pred_ph = st.empty()
lock_ph = st.empty()

# =========================================================
# Webcam Loop
# =========================================================
if run:
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # -------------------------
        # YOLO Hand Detection
        # -------------------------
        results = hand_detector(frame, conf=0.45, verbose=False)
        hands = []

        for r in results:
            if r.boxes:
                for b in r.boxes:
                    det_conf = float(b.conf[0])
                    if det_conf < 0.50:
                        continue

                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                    # square crop with padding
                    bw, bh = x2 - x1, y2 - y1
                    side = max(bw, bh)
                    pad = int(side * 0.25)

                    xmin = max(0, x1 - pad)
                    ymin = max(0, y1 - pad)
                    xmax = min(w, x2 + pad)
                    ymax = min(h, y2 + pad)

                    crop = frame[ymin:ymax, xmin:xmax]
                    hands.append((crop, det_conf, (xmin, ymin, xmax, ymax)))

        # If no hands
        if not hands:
            frame_ph.image(frame, channels="BGR")
            continue

        # --------------------------------
        # Classify EACH hand separately
        # Use the one with BEST confidence
        # --------------------------------
        best_label = None
        best_conf = 0
        best_box = None

        for crop, det_conf, box in hands:
            if crop.size == 0:
                continue

            resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (96,96))
            inp = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()

            out = interpreter.get_tensor(output_details[0]['index'])[0]
            idx = np.argmax(out)
            model_conf = float(out[idx])
            label = labels[idx]

            combined = model_conf * det_conf   # simple & accurate

            if combined > best_conf:
                best_conf = combined
                best_label = label
                best_box = box

        # Draw best hand bounding box
        if best_box:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,255),2)

        # -------------------------
        # Prediction Smoothing
        # -------------------------
        if best_label:
            pred_queue.append((best_label, best_conf))
            ema_scores[best_label] = ema_alpha * best_conf + (1 - ema_alpha) * ema_scores.get(best_label, 0)

        # choose best via EMA
        display_label = None
        if ema_scores:
            display_label = max(ema_scores, key=lambda k: ema_scores[k])

        # fallback = majority vote
        if not display_label and pred_queue:
            votes = {}
            for l, c in pred_queue:
                votes[l] = votes.get(l, 0) + c
            display_label = max(votes, key=votes.get)

        # -------------------------
        # Locking
        # -------------------------
        if locked_letter:
            lock_ph.markdown(f"### 🔒 Locked: **{locked_letter}**")
            lock_counter -= 1
            if lock_counter <= 0:
                locked_letter = None
                lock_ph.empty()
        else:
            if display_label:
                strong = sum(1 for l,c in pred_queue if l == display_label and c > 0.80)
                if strong > 10:
                    locked_letter = display_label
                    lock_counter = lock_frames

                pred_ph.markdown(f"### Prediction: **{display_label}**")

        frame_ph.image(frame, channels="BGR")

    cap.release()
