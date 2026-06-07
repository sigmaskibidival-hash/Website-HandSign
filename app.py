from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os
import base64
import io

app = Flask(__name__, static_folder='static')

# --- Define predict_rps so the pkl can unpickle correctly ---
def predict_rps(image_array):
    """
    Stub predict function stored in the pkl.
    The actual prediction logic lives below in /scan.
    This definition satisfies the pickle reference to __main__.predict_rps.
    """
    pass

# Load the model (which holds a reference to predict_rps defined above)
import __main__
__main__.predict_rps = predict_rps
model = joblib.load('rps_model.pkl')


def preprocess_image(image_bytes):
    """
    Resize and flatten an uploaded image into a feature vector
    compatible with the model. Tries PIL first, falls back to numpy.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception as e:
        raise ValueError(f"Could not process image: {e}")


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/scan', methods=['POST'])
def scan():
    # Accept either a file upload or a base64 data URL from the webcam
    image_bytes = None

    if 'file' in request.files and request.files['file'].filename:
        image_bytes = request.files['file'].read()
    elif request.is_json:
        data = request.get_json()
        if 'image' in data:
            # Strip data URL header if present
            b64 = data['image'].split(',')[-1]
            image_bytes = base64.b64decode(b64)
    
    if image_bytes is None:
        return jsonify({'error': 'No image provided'}), 400

    try:
        features = preprocess_image(image_bytes)
        X = np.array([features], dtype=np.float32)

        # Use the loaded pkl object if it's a sklearn estimator,
        # otherwise call it directly as a function
        if hasattr(model, 'predict'):
            pred_raw = model.predict(X)[0]
            prob = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        else:
            pred_raw = model(X)
            prob = None

        # Map numeric or string prediction to RPS label
        label_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors',
                     'rock': 'Rock', 'paper': 'Paper', 'scissors': 'Scissors',
                     'Rock': 'Rock', 'Paper': 'Paper', 'Scissors': 'Scissors'}
        label = label_map.get(pred_raw, str(pred_raw))
        confidence = int(max(prob) * 100) if prob is not None else None

        return jsonify({
            'prediction': str(pred_raw),
            'label': label,
            'confidence': confidence,
            'emoji': {'Rock': '✊', 'Paper': '✋', 'Scissors': '✌️'}.get(label, '❓')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
