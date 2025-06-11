from flask import Flask, render_template, Response, send_from_directory, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import datetime
import csv
import platform
import subprocess
import smtplib
from email.message import EmailMessage
import threading
import random
from collections import deque

# === CONFIGURATION ===
BELT_SPEED = 3.5  # Initial speed in m/s
MODEL_PATH = 'modelcnn2.h5'
VIDEO_PATH = 'grp img 2.mp4'
SNAPSHOT_DIR = 'static/snapshots'
LOG_PATH = 'logs/anomalies.csv'
EMAIL_ALERT = False
MAX_ANOMALIES = 100  # Max anomalies to keep in memory
BELT_LENGTH = 50  # Virtual belt length in meters

# === INIT FLASK + SOCKETIO ===
app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app, async_mode='eventlet')

# === CREATE DIRECTORIES ===
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("static/belt_positions", exist_ok=True)

# === INIT CSV ===
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "prediction", "snapshot_path", "position", "belt_speed"])

# === GLOBAL VARIABLES ===
model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_lock = threading.Lock()
output_frame = None
latest_score = 0.0
alerts = deque(maxlen=MAX_ANOMALIES)
belt_speed = BELT_SPEED
start_time = datetime.datetime.now()
anomaly_positions = {}  # Track anomalies by position

# === UTILITY FUNCTIONS ===
def save_snapshot(frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{SNAPSHOT_DIR}/{timestamp}.jpg"
    cv2.imwrite(path, frame)
    return path

def log_anomaly(pred, snapshot_path, position, speed):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, pred, snapshot_path, f"{position:.2f}", f"{speed:.2f}"])
    
    anomaly_data = {
        "time": timestamp, 
        "prediction": float(pred), 
        "snapshot_path": snapshot_path,
        "position": float(position),
        "belt_speed": float(speed),
        "id": f"anom-{timestamp.replace(':', '').replace(' ', '-')}"
    }
    
    # Store position data for belt visualization
    anomaly_positions[position] = anomaly_data
    
    return anomaly_data

def send_email_notification(subject, body):
    if not EMAIL_ALERT:
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = "your_email@example.com"
    msg['To'] = "recipient@example.com"
    try:
        with smtplib.SMTP('smtp.example.com', 587) as smtp:
            smtp.starttls()
            smtp.login('your_email@example.com', 'your_password')
            smtp.send_message(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")

def send_system_notification(message):
    try:
        if platform.system() == "Linux":
            subprocess.run(['notify-send', message])
        elif platform.system() == "Windows":
            subprocess.run(['msg', '*', message])
        else:
            print(f"Notification: {message}")
    except:
        print("System notification failed")

def generate_belt_image(position, anomalies):
    """Generate a belt visualization image with anomaly markers"""
    width = 1000
    height = 100
    belt_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw belt
    belt_img[:, :, :] = (200, 200, 200)  # Gray belt
    
    # Draw belt edges
    cv2.line(belt_img, (0, 0), (width, 0), (100, 100, 100), 2)
    cv2.line(belt_img, (0, height-1), (width, height-1), (100, 100, 100), 2)
    
    # Draw rollers
    roller_spacing = 50
    for x in range(0, width, roller_spacing):
        cv2.rectangle(belt_img, (x, 20), (x+10, height-20), (150, 150, 150), -1)
    
    # Draw current position indicator
    current_pos = int((position % BELT_LENGTH) / BELT_LENGTH * width)
    cv2.line(belt_img, (current_pos, 0), (current_pos, height), (0, 0, 255), 2)
    
    # Draw anomalies
    for pos, anomaly in anomalies.items():
        if abs(position - pos) < BELT_LENGTH:  # Only show anomalies within belt length
            x = int((pos % BELT_LENGTH) / BELT_LENGTH * width)
            severity = "high" if anomaly['prediction'] < 0.3 else "medium"
            color = (0, 0, 255) if severity == "high" else (0, 165, 255)
            cv2.circle(belt_img, (x, height//2), 10, color, -1)
            cv2.putText(belt_img, f"{anomaly['prediction']:.2f}", (x-15, height//2-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    # Save belt image
    path = f"static/belt_positions/belt_{position:.2f}.jpg"
    cv2.imwrite(path, belt_img)
    return path

# === VIDEO PROCESSING ===
def process_video():
    global latest_score, output_frame, belt_speed, alerts

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        resized = cv2.resize(frame, (64, 64))
        frame_array = img_to_array(resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        prediction = model.predict(frame_array)[0][0]
        latest_score = float(prediction)

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = cap.get(cv2.CAP_PROP_FPS)
        time_elapsed = frame_number / fps
        position_m = belt_speed * time_elapsed

        # Generate belt visualization
        belt_img_path = generate_belt_image(position_m, anomaly_positions)

        if prediction < 0.5:
            snapshot_path = save_snapshot(frame)
            anomaly_data = log_anomaly(prediction, snapshot_path, position_m, belt_speed)

            alert = {
                'time': datetime.datetime.now().strftime("%H:%M:%S"),
                'score': float(prediction),
                'severity': "High" if prediction < 0.2 else "Moderate",
                'color': "#e74c3c" if prediction < 0.2 else "#f39c12",
                'position': f"{position_m:.2f} m",
                'belt_speed': f"{belt_speed:.2f} m/s",
                'snapshot_path': snapshot_path,
                'belt_img_path': belt_img_path,
                'id': anomaly_data['id']
            }
            alerts.append(alert)
            socketio.emit('new_alert', alert)

            send_system_notification("⚠️ Anomaly detected on the belt!")
            if EMAIL_ALERT:
                send_email_notification("Belt Anomaly", f"Anomaly detected with score {prediction:.2f}")

            label = "⚠️ Anomaly"
            color = (0, 0, 255)
        else:
            label = "✅ Normal"
            color = (0, 255, 0)

        # Add overlay to frame
        cv2.putText(frame, f"Position: {position_m:.2f} m", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        with frame_lock:
            _, buffer = cv2.imencode('.jpg', frame)
            output_frame = buffer.tobytes()

# === VIDEO STREAM ===
def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')

# === FLASK ROUTES ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot/<filename>')
def snapshot_image(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

@app.route('/belt_image/<filename>')
def belt_image(filename):
    return send_from_directory('static/belt_positions', filename)

@app.route('/download_logs')
def download_logs():
    return send_from_directory('logs', 'anomalies.csv', as_attachment=True)

@app.route('/data')
def get_data():
    anomalies = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            anomalies = list(reader)[-100:]  # Return last 100 anomalies
    
    uptime = str(datetime.datetime.now() - start_time).split('.')[0]
    
    return jsonify({
        "latest_score": latest_score,
        "alerts": list(alerts),
        "belt_speed": belt_speed,
        "uptime": uptime,
        "belt_length": BELT_LENGTH,
        "anomaly_positions": list(anomaly_positions.values())[-10:]  # Recent anomalies for belt
    })

@app.route('/anomaly/<anomaly_id>')
def get_anomaly_details(anomaly_id):
    for anomaly in reversed(list(anomaly_positions.values())):
        if anomaly['id'] == anomaly_id:
            return jsonify(anomaly)
    return jsonify({"error": "Anomaly not found"}), 404

@app.route('/update_speed', methods=['POST'])
def update_speed():
    global belt_speed
    try:
        new_speed = float(request.form['speed'])
        belt_speed = new_speed
        return jsonify({"status": "success", "new_speed": belt_speed})
    except:
        return jsonify({"status": "error", "message": "Invalid speed value"}), 400

@app.route('/generate_report')
def generate_report():
    from fpdf import FPDF
    import os, csv
    from flask import send_from_directory

    anomalies = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            anomalies = list(reader)

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, 'Anomaly Detection Report', 0, 1, 'C')
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    for anomaly in anomalies:
        pdf.cell(0, 10, f"Time: {anomaly['timestamp']}", ln=True)
        pdf.cell(0, 10, f"Score: {anomaly['prediction']}", ln=True)
        pdf.cell(0, 10, f"Position: {anomaly.get('position', 'N/A')}", ln=True)
        
        snapshot_path = os.path.join(SNAPSHOT_DIR, anomaly['snapshot_path'].split('/')[-1])
        if os.path.exists(snapshot_path):
            pdf.image(snapshot_path, w=60)
        pdf.ln(10)

    report_path = "logs/report.pdf"
    pdf.output(report_path)
    return send_from_directory("logs", "report.pdf", as_attachment=True)

# === RUN APPLICATION ===
if __name__ == '__main__':
    threading.Thread(target=process_video, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)