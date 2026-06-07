from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os
import base64
import io

app = Flask(__name__, static_folder='static')

# ---------------------------------------------------------------------------
# MediaPipe hand landmark setup
# ---------------------------------------------------------------------------
import mediapipe as mp

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# ---------------------------------------------------------------------------
# RPS classification from landmarks
# ---------------------------------------------------------------------------

def _finger_extended(landmarks, tip_id, pip_id):
    """Return True if a finger is extended (tip above PIP joint in image coords)."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def classify_rps(landmarks):
    """
    Classify Rock / Paper / Scissors from 21 MediaPipe hand landmarks.

    Finger tip / PIP landmark indices:
      Index  : tip=8,  pip=6
      Middle : tip=12, pip=10
      Ring   : tip=16, pip=14
      Pinky  : tip=20, pip=18
      Thumb  : tip=4,  ip=3  (uses x-axis comparison)

    Returns (label, confidence_pct).
    """
    lm = landmarks

    index  = _finger_extended(lm, 8,  6)
    middle = _finger_extended(lm, 12, 10)
    ring   = _finger_extended(lm, 16, 14)
    pinky  = _finger_extended(lm, 20, 18)

    # Thumb: compare x position of tip vs IP joint (works for right hand facing camera)
    thumb = abs(lm[4].x - lm[3].x) > 0.04

    fingers_up = sum([index, middle, ring, pinky])

    # --- Rules ---
    if fingers_up == 0:                          # all fingers curled
        label, conf = 'Rock', 92
    elif fingers_up == 4:                        # all fingers open
        label, conf = 'Paper', 95
    elif index and middle and not ring and not pinky:  # V sign
        label, conf = 'Scissors', 93
    elif index and not middle and not ring and not pinky:  # pointing
        label, conf = 'Scissors', 75
    elif fingers_up >= 3:
        label, conf = 'Paper', 70
    elif fingers_up == 1:
        label, conf = 'Rock', 65
    else:
        label, conf = 'Rock', 55   # fallback

    return label, conf


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_rgb_image(image_bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/scan', methods=['POST'])
def scan():
    image_bytes = None

    if 'file' in request.files and request.files['file'].filename:
        image_bytes = request.files['file'].read()
    elif request.is_json:
        data = request.get_json()
        if 'image' in data:
            b64 = data['image'].split(',')[-1]
            image_bytes = base64.b64decode(b64)

    if image_bytes is None:
        return jsonify({'error': 'No image provided'}), 400

    try:
        img_rgb = load_rgb_image(image_bytes)
    except Exception as e:
        return jsonify({'error': f'Could not decode image: {e}'}), 400

    result = hands_detector.process(img_rgb)

    if not result.multi_hand_landmarks:
        return jsonify({'error': 'No hand detected — make sure your hand is clearly visible'}), 200

    landmarks = result.multi_hand_landmarks[0].landmark
    label, confidence = classify_rps(landmarks)

    return jsonify({
        'label': label,
        'confidence': confidence,
        'emoji': {'Rock': '✊', 'Paper': '✋', 'Scissors': '✌️'}[label],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
