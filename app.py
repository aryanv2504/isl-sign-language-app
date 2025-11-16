import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(
    page_title="ISL Sign Language App",
    layout="wide",
    page_icon="🤟"
)

# ===============================
# Custom CSS (simpler & centered)
# ===============================
st.markdown("""
<style>
    .main {
        background: #0d0d0d;
        color: white;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #00eaff;
        text-shadow: 0 0 15px #00eaff;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 20px;
    }
    .box {
        background: rgba(0,255,255,0.08);
        border: 2px solid #00eaff;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        width: 320px;
        box-shadow: 0 0 15px #00eaff;
        margin-top: 12px;
    }
    .locked-box {
        background: rgba(0,255,0,0.08);
        border: 2px solid #00ff88;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        width: 320px;
        box-shadow: 0 0 15px #00ff88;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Title
# ===============================
st.markdown("<div class='title'>ISL Sign Language App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time ISL Letter Recognition</div>", unsafe_allow_html=True)

# ===============================
# Sidebar Control
# ===============================
st.sidebar.title("⚙️ Controls")
run = st.sidebar.checkbox("Start Webcam")

# ===============================
# Load ML Model
# ===============================
labels = [line.strip() for line in open("labels.txt")]

interpreter = tf.lite.Interpreter(model_path="isl_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

# Prediction smoothing
pred_queue = deque(maxlen=50)
locked_letter = None
lock_frames = 30
lock_counter = 0

# ===============================
# SINGLE CENTER COLUMN (IMPORTANT)
# ===============================
root = st.container()
with root:
    col = st.columns([1, 2, 1])[1]   # center column only

    with col:
        camera_placeholder = st.empty()
        prediction_placeholder = st.empty()
        locked_placeholder = st.empty()

# ===============================
# Webcam Loop
# ===============================
if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            all_x, all_y = [], []
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                for lm in handLms.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)

            xmin, xmax = int(min(all_x)*w), int(max(all_x)*w)
            ymin, ymax = int(min(all_y)*h), int(max(all_y)*h)

            pad = 25
            xmin, ymin = max(0, xmin-pad), max(0, ymin-pad)
            xmax, ymax = min(w, xmax+pad), min(h, ymax+pad)

            crop = frame[ymin:ymax, xmin:xmax]

            if crop.size != 0:
                resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (96, 96))
                input_data = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                out = interpreter.get_tensor(output_details[0]['index'])[0]
                idx = np.argmax(out)
                conf = out[idx]
                label = labels[idx].upper()

                pred_queue.append((label, conf))

                # ============= LOCKING LOGIC FIXED =============
                if locked_letter:
                    locked_placeholder.markdown(
                        f"<div class='locked-box'><h2>Locked: {locked_letter}</h2></div>",
                        unsafe_allow_html=True
                    )
                    prediction_placeholder.empty()  # remove prediction box

                    lock_counter -= 1
                    if lock_counter <= 0:
                        locked_letter = None
                        locked_placeholder.empty()

                else:
                    counts = {}
                    for l, c in pred_queue:
                        if c > 0.90:
                            counts[l] = counts.get(l, 0) + 1

                    if counts:
                        best = max(counts, key=counts.get)
                        if counts[best] > 30:
                            locked_letter = best
                            lock_counter = lock_frames

                    # SHOW ONLY ONE prediction box
                    prediction_placeholder.markdown(
                        f"<div class='box'><h3>Prediction: {label}</h3></div>",
                        unsafe_allow_html=True
                    )

        # CENTERED CAMERA OUTPUT
        camera_placeholder.image(frame, channels="BGR")

    cap.release()
