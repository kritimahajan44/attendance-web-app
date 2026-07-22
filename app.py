import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
import numpy as np
import cv2
import csv
from flask import Flask, render_template_string, request, jsonify
from deepface import DeepFace
from datetime import datetime

app = Flask(__name__)

# Directory setup
DB_PATH = r"C:\Users\dell\Documents\python\faces_db"
CSV_PATH = r"C:\Users\dell\Documents\python\attendance.csv"

os.makedirs(DB_PATH, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        f.write("Name,Date,Time\n")

def mark_attendance(name):
    with open(CSV_PATH, 'r') as f:
        lines = f.readlines()
        present_names = [line.split(',')[0] for line in lines]
        
    if name not in present_names:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f"{name},{date_str},{time_str}\n")
        return f"✅ Attendance marked for {name}!"
    return f"ℹ️ {name} is already marked present today."

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Attendance Kiosk</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background: #f4f7f6; padding: 20px; }
        #video { border: 3px solid #333; border-radius: 8px; width: 480px; height: 360px; }
        .btn { padding: 12px 24px; margin: 10px; font-size: 16px; font-weight: bold; cursor: pointer; border: none; border-radius: 5px; text-decoration: none; display: inline-block; }
        .btn-scan { background: #28a745; color: white; }
        .btn-reg { background: #007bff; color: white; }
        .btn-view { background: #17a2b8; color: white; }
        #status { font-weight: bold; font-size: 18px; margin-top: 20px; color: #333; }
    </style>
</head>
<body>
    <h2>📷 Face Recognition Attendance Kiosk</h2>
    <video id="video" autoplay></video><br>
    <button class="btn btn-scan" onclick="scanFace()">Mark Attendance</button>
    <button class="btn btn-reg" onclick="registerFace()">Register New Person</button>
    <a href="/attendance" target="_blank" class="btn btn-view">📋 View Attendance Log</a>
    <div id="status">Ready</div>

    <script>
        const video = document.getElementById('video');
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => video.srcObject = stream)
            .catch(err => alert("Webcam access error: " + err));

        function captureFrame() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            return canvas.toDataURL('image/jpeg');
        }

        async function scanFace() {
            document.getElementById('status').innerText = "Analyzing...";
            const image = captureFrame();
            const res = await fetch('/scan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ image })
            });
            const data = await res.json();
            document.getElementById('status').innerText = data.message;
        }

        async function registerFace() {
            const name = prompt("Enter person's full name:");
            if (!name) return;
            document.getElementById('status').innerText = "Registering...";
            const image = captureFrame();
            const res = await fetch('/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name, image })
            });
            const data = await res.json();
            document.getElementById('status').innerText = data.message;
        }
    </script>
</body>
</html>
"""

ATTENDANCE_TABLE_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Register</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 30px; background: #f4f7f6; text-align: center; }
        table { margin: 0 auto; border-collapse: collapse; width: 60%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        th, td { padding: 12px 15px; text-align: center; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; font-size: 18px; }
        tr:hover { background-color: #f1f1f1; }
        .back-btn { display: inline-block; margin-bottom: 20px; text-decoration: none; color: #007bff; font-weight: bold; font-size: 16px; }
    </style>
</head>
<body>
    <h2>📋 Attendance Log</h2>
    <a class="back-btn" href="/">← Back to Kiosk</a>
    <table>
        {% for row in rows %}
        <tr>
            {% if loop.first %}
                {% for col in row %}<th>{{ col }}</th>{% endfor %}
            {% else %}
                {% for col in row %}<td>{{ col }}</td>{% endfor %}
            {% endif %}
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/attendance')
def attendance():
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
    return render_template_string(ATTENDANCE_TABLE_PAGE, rows=rows)

def decode_image(data_url):
    encoded_data = data_url.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/scan', methods=['POST'])
def scan():
    frame = decode_image(request.json['image'])
    db_files = [f for f in os.listdir(DB_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not db_files:
        return jsonify({"message": "⚠️ Database is empty. Please register first using the blue button."})
    
    try:
        dfs = DeepFace.find(img_path=frame, db_path=DB_PATH, detector_backend='skip', enforce_detection=False, silent=True)
        if len(dfs) > 0 and not dfs[0].empty:
            matched_file = dfs[0].iloc[0]['identity']
            person_name = os.path.basename(matched_file).rsplit('.', 1)[0]
            msg = mark_attendance(person_name)
            return jsonify({"message": msg})
        return jsonify({"message": "⚠️ Face unrecognized. Please register first."})
    except Exception as e:
        return jsonify({"message": f"Error during scan: {str(e)}"})

@app.route('/register', methods=['POST'])
def register():
    name = request.json.get('name', '').strip()
    frame = decode_image(request.json['image'])
    if name:
        save_path = os.path.join(DB_PATH, f"{name}.jpg")
        cv2.imwrite(save_path, frame)
        
        pkl_files = [f for f in os.listdir(DB_PATH) if f.endswith('.pkl')]
        for pkl in pkl_files:
            try: os.remove(os.path.join(DB_PATH, pkl))
            except: pass
            
        msg = mark_attendance(name)
        return jsonify({"message": f"✨ Successfully registered {name}! {msg}"})
    return jsonify({"message": "Cancelled registration."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)