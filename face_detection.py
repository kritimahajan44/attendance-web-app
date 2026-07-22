import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog

# Hide default blank tkinter root window
root = tk.Tk()
root.withdraw()

# 1. Setup paths
db_path = r"C:\Users\dell\Documents\python\faces_db"
csv_path = r"C:\Users\dell\Documents\python\attendance.csv"

os.makedirs(db_path, exist_ok=True)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        f.write("Name,Date,Time\n")
    print(f"📄 Attendance log initialized at: {csv_path}")

def mark_attendance(name):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        present_names = [line.split(',')[0] for line in lines]
        
    if name not in present_names:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        with open(csv_path, 'a') as f:
            f.write(f"{name},{date_str},{time_str}\n")
        print(f"✅ Attendance marked for {name} at {time_str}!")

def register_user(frame):
    new_name = simpledialog.askstring(title="Register New User", prompt="Enter person's name:")
    if new_name and new_name.strip():
        new_name = new_name.strip()
        save_path = os.path.join(db_path, f"{new_name}.jpg")
        cv2.imwrite(save_path, frame)
        
        # Clear cache pickles so new image indexes immediately
        pkl_files = [f for f in os.listdir(db_path) if f.endswith('.pkl')]
        for pkl in pkl_files:
            try:
                os.remove(os.path.join(db_path, pkl))
            except Exception:
                pass
            
        print(f"✨ Successfully registered {new_name}!")
        mark_attendance(new_name)
    else:
        print("⚠️ Registration cancelled.")

# 2. Start Live Video Stream
print("\n==================================================")
print("     ATTENDANCE SYSTEM ACTIVE")
print("==================================================")
print(" * Press 'r' on camera window to identify/register.")
print(" * Press 'q' on camera window to exit.\n")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab camera frame.")
        break

    cv2.imshow('Attendance System', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        print("\n🔍 Analyzing face...")
        
        # Check if database is empty first
        db_files = [f for f in os.listdir(db_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not db_files:
            print("📁 Database is empty! Triggering first user registration...")
            register_user(frame)
        else:
            try:
                dfs = DeepFace.find(img_path=frame, db_path=db_path, detector_backend='opencv', enforce_detection=False, silent=True)
                
                if len(dfs) > 0 and not dfs[0].empty:
                    matched_file = dfs[0].iloc[0]['identity']
                    person_name = os.path.basename(matched_file).rsplit('.', 1)[0]
                    print(f"✨ Match Found: {person_name}")
                    mark_attendance(person_name)
                else:
                    print("⚠️ Unrecognized face.")
                    register_user(frame)
            except Exception as e:
                print(f"⚠️ Error during scan ({e}). Launching registration...")
                register_user(frame)

    elif key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("\nSystem closed successfully.")