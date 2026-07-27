import os
import cv2
import base64
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "faces_db")
CSV_PATH = os.path.join(BASE_DIR, "attendance.csv")

app = Flask(__name__, template_folder='.')

# Create paths and initial CSV structure
os.makedirs(DB_PATH, exist_ok=True)
if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(CSV_PATH, index=False)

# Load OpenCV's built-in Haar Cascade Face Detector
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def decode_image(data_url):
    """Converts base64 string from browser into an OpenCV BGR image matrix."""
    try:
        encoded_data = data_url.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def extract_face(frame):
    """Detects and returns cropped grayscale face image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    return cv2.resize(face_crop, (200, 200))

def mark_attendance(name):
    """Logs student name and timestamp to attendance.csv."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    df = pd.read_csv(CSV_PATH)
    already_marked = ((df["Name"] == name) & (df["Date"] == date_str)).any()

    if already_marked:
        return f"ℹ️ Attendance for {name} is already marked today!"
    else:
        new_row = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        return f"✅ Attendance successfully marked for {name} at {time_str}!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json or {}
        name = data.get('name', '').strip()
        image_data = data.get('image', '')

        if not name:
            return jsonify({"message": "⚠️ Please enter a name first!"})
        if not image_data:
            return jsonify({"message": "⚠️ No image captured!"})

        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"message": "⚠️ Failed to decode image payload!"})

        face_img = extract_face(frame)
        if face_img is None:
            return jsonify({"message": "⚠️ No face detected. Position yourself clearly in front of the camera!"})

        file_path = os.path.join(DB_PATH, f"{name}.jpg")
        cv2.imwrite(file_path, face_img)

        return jsonify({"message": f"✅ Registered {name} successfully!"})

    except Exception as e:
        return jsonify({"message": f"Error during registration: {str(e)}"})

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.json or {}
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({"message": "⚠️ No image captured!"})

        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"message": "⚠️ Invalid image payload!"})

        face_img = extract_face(frame)
        if face_img is None:
            return jsonify({"message": "⚠️ No face detected in frame. Look directly at camera."})

        registered_files = [f for f in os.listdir(DB_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not registered_files:
            return jsonify({"message": "⚠️ Database is empty. Please register first!"})

        # Match captured face against registered template faces using Mean Square Error / Template Matching
        best_match = None
        min_diff = float('inf')

        for file in registered_files:
            ref_path = os.path.join(DB_PATH, file)
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                continue

            ref_img = cv2.resize(ref_img, (200, 200))
            diff = np.mean((face_img.astype("float") - ref_img.astype("float")) ** 2)

            if diff < min_diff:
                min_diff = diff
                best_match = os.path.splitext(file)[0]

        # Difference threshold to decide face match
        if min_diff < 4500 and best_match:
            msg = mark_attendance(best_match)
            return jsonify({"message": msg})
        
        return jsonify({"message": "⚠️ Face unrecognized. Please register first."})

    except Exception as e:
        return jsonify({"message": f"Error during scan: {str(e)}"})

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            return jsonify(df.to_dict(orient='records'))
        return jsonify([])
    except Exception:
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)