from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
from keras.models import model_from_json
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import csv
import os
import sqlite3
import time

app = Flask(__name__)

# Initialize emotion detection model
try:
    json_file = open("emotiondetector.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("emotiondetector.h5")
    print("Emotion detection model loaded successfully")
except Exception as e:
    print(f"Error loading emotion detection model: {e}")
    model = None

# Load the face cascade classifier
try:
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    print("Face cascade classifier loaded successfully")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    face_cascade = None

# Function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize with a default emotion
prediction_label = "neutral"

# Create a dictionary mapping numerical labels to emotions
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def detect_emotion():
    global prediction_label
    
    try:
        # Try to open webcam
        webcam = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not webcam.isOpened():
            print("Camera not available - using placeholder")
            # Create a placeholder frame for cloud deployment
            while True:
                frame = np.zeros((600, 1028, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available in cloud deployment", (200, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Using '{prediction_label}' as default emotion", (200, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Refresh for updated recommendations", (200, 350), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Every 10 seconds, cycle through emotions for demo purposes in cloud
                if os.environ.get('RENDER') == 'true':
                    current_time = int(time.time())
                    emotion_index = (current_time // 10) % 7  # Change emotion every 10 seconds
                    prediction_label = emotion_labels[emotion_index]
                
                # Convert frame to bytes
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Add a small delay to reduce CPU usage
                time.sleep(0.1)
        
        # If camera is available, proceed with normal emotion detection
        while True:
            ret, frame = webcam.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1028, 600))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                
                # Resize image and extract features
                resized_image = cv2.resize(roi_gray, (48, 48))
                img = extract_features(resized_image)
                
                # Predict emotion
                pred = model.predict(img)
                prediction_label = emotion_labels[pred.argmax()]
                
                # Draw rectangle and display emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Add text showing current detected emotion
            cv2.putText(frame, f"Detected Emotion: {prediction_label}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert frame to bytes
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        # Create a placeholder frame when an error occurs
        while True:
            frame = np.zeros((600, 1028, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera error: {str(e)[:50]}", (100, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Using default emotion for recommendations", (100, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert frame to bytes
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.1)

@app.route('/get_emotion')                
def get_emotion():
    global prediction_label
    # Ensure we never return None
    if prediction_label is None:
        prediction_label = "neutral"
    return prediction_label

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Iot(db.Model):
    sno = db.Column(db.Integer(), primary_key=True)
    email = db.Column(db.String(), nullable=False)
    name = db.Column(db.String(), nullable=False)
    EmotionDetected = db.Column(db.String(), nullable=False)
    SongRecommended = db.Column(db.String(), nullable=False)

class Login(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

with app.app_context():
    db.create_all()

@app.route('/')
def ind():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == "POST":
        EmotionDetected = request.form['EmotionDetected']
        SongRecommended = request.form['SongRecommended']
        name = request.form['name']
        email = request.form['email']
        iot = Iot(email=email, EmotionDetected=EmotionDetected, SongRecommended=SongRecommended, name=name)
        db.session.add(iot)
        db.session.commit()
    alldata = Iot.query.all()
    return render_template('form.html', alldata=alldata)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_login = Login(username=username, password=password)
        db.session.add(new_login)
        db.session.commit()
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    entered_fname = request.form['username']
    entered_pword = request.form['password']

    if authenticate(entered_fname, entered_pword):
        return render_template('home.html')
    else:
        return "Access Denied"

def authenticate(username, password):
    conn = sqlite3.connect('instance/project.db')
    cursor = conn.cursor()

    # Fix column name from 'fullname' to 'username'
    cursor.execute('''
   CREATE TABLE IF NOT EXISTS Login (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       username TEXT,
       password TEXT
   )
''')

    cursor.execute("SELECT * FROM Login WHERE username=? AND password=?", (username, password))
    usinger = cursor.fetchone()

    conn.close()

    return usinger is not None

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/read_csv')
def read_csv():
    global prediction_label
    
    # Ensure prediction_label is not None
    if prediction_label is None:
        prediction_label = "neutral"
    
    # Define expected emotion CSV files
    expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # If prediction_label is not in our expected list, default to neutral
    if prediction_label not in expected_emotions:
        prediction_label = "neutral"
        
    songs = []
    emotionfile = f"{prediction_label}.csv"
    
    try:
        # Try to read the CSV file for the current emotion
        print(f"Attempting to read CSV file: {emotionfile}")
        with open(emotionfile, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                songs.append({
                    'Name': row.get('Name', 'Unknown'),
                    'Album': row.get('Album', 'Unknown'),
                    'Artist': row.get('Artist', 'Unknown'),
                    'Link': row.get('Link', '#'),
                    'Image': row.get('Image', '/static/styles/assets/logo.png')
                })
    except FileNotFoundError:
        print(f"CSV file not found: {emotionfile}")
        # If the CSV file doesn't exist, try to use neutral.csv as fallback
        try:
            with open('neutral.csv', 'r', newline='', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    songs.append({
                        'Name': row.get('Name', 'Unknown'),
                        'Album': row.get('Album', 'Unknown'),
                        'Artist': row.get('Artist', 'Unknown'),
                        'Link': row.get('Link', '#'),
                        'Image': row.get('Image', '/static/styles/assets/logo.png')
                    })
            print("Using neutral.csv as fallback")
        except FileNotFoundError:
            # If neutral.csv also doesn't exist, create default songs
            songs = [
                {'Name': 'Happiness', 'Album': 'Default Album', 'Artist': 'Default Artist', 
                 'Link': 'https://open.spotify.com/track/2QjOHCTQ1Jl3zawyYOpxh6', 'Image': '/static/styles/assets/logo.png'},
                {'Name': 'Calm', 'Album': 'Default Album', 'Artist': 'Default Artist', 
                 'Link': 'https://open.spotify.com/track/0V3wPSX9ygBnCm8psDIegu', 'Image': '/static/styles/assets/logo.png'},
                {'Name': 'Energy', 'Album': 'Default Album', 'Artist': 'Default Artist', 
                 'Link': 'https://open.spotify.com/track/7dt6x5M1jzdTEt8oCbisTK', 'Image': '/static/styles/assets/logo.png'}
            ]
            print("Using hardcoded default songs")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # For any other error, create default songs
        songs = [
            {'Name': 'Error Reading Songs', 'Album': 'Please check file format', 'Artist': 'System', 
             'Link': '#', 'Image': '/static/styles/assets/logo.png'}
        ]
    
    return jsonify(songs)

@app.route('/play/<path:filename>')
def play(filename):
    audio_folder = 'audio'
    file_path = os.path.join(audio_folder, filename)
    try:
        return send_file(file_path, as_attachment=False)
    except FileNotFoundError:
        return "Audio file not found", 404

@app.route('/songs')
def songs():
    return render_template('songs.html')

# Add a health check endpoint
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "current_emotion": prediction_label,
        "camera_available": False if os.environ.get('RENDER') == 'true' else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)