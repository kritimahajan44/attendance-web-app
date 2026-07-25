import os
import glob
import gc
import cv2
import numpy as np
import pandas as pd
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Absolute paths to avoid directory relative-path issues on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "faces_db")
CSV_PATH = os.path.join(BASE_DIR, "attendance.csv")

# Ensure required directory and CSV exist
os.makedirs(DB_PATH, exist_ok=True)

if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(CSV_PATH, index=False)

def decode_image(data_url):
    try:
        encoded_data = data_url.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return None

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
        else:
            df = pd.read_csv(CSV_PATH)

        todays_entries = df[(df['Name'] == name) & (df['Date'] == date_str)]

        if not todays_entries.empty:
            return f"ℹ️ {name} is already marked present for today!"

        new_entry = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        return f"✅ Attendance marked for {name} at {time_str}!"
    except Exception as e:
        return f"⚠️ Failed to update attendance log: {str(e)}"

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Template Error: {str(e)}", 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json or {}
        name = data.get('name', '').strip()
        image_data = data.get('image', '')

        if not name or not image_data:
            return jsonify({"message": "⚠️ Name and image are required!"})

        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"message": "⚠️ Invalid image payload!"})

        file_path = os.path.join(DB_PATH, f"{name}.jpg")
        cv2.imwrite(file_path, frame)

        # Remove old DeepFace pkl caches to re-index faces immediately
        for pkl in glob.glob(os.path.join(DB_PATH, "*.pkl")):
            try:
                os.remove(pkl)
            except Exception:
                pass

        return jsonify({"message": f"✅ {name} registered successfully!"})
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
            return jsonify({"message": "⚠️ Invalid image format!"})

        db_files = [f for f in os.listdir(DB_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not db_files:
            return jsonify({"message": "⚠️ Database is empty. Please register first using the blue button."})

        # Remove stale pickle cache files
        for pkl in glob.glob(os.path.join(DB_PATH, "*.pkl")):
            try:
                os.remove(pkl)
            except Exception:
                pass

        # Perform fast DeepFace match using OpenCV backend for stability on free Render instances
        dfs = DeepFace.find(
            img_path=frame, 
            db_path=DB_PATH, 
            model_name='Facenet', 
            detector_backend='opencv', 
            enforce_detection=False, 
            silent=True
        )

        gc.collect()

        if len(dfs) > 0 and not dfs[0].empty:
            matched_file = dfs[0].iloc[0]['identity']
            person_name = os.path.basename(matched_file).rsplit('.', 1)[0]
            msg = mark_attendance(person_name)
            return jsonify({"message": msg})
        
        return jsonify({"message": "⚠️ Face unrecognized. Please register first."})

    except Exception as e:
        gc.collect()
        return jsonify({"message": f"Error during scan: {str(e)}"})

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            return jsonify([])
        df = pd.read_csv(CSV_PATH)
        records = df.to_dict(orient='records')
        return jsonify(records)
    except Exception as e:
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)