from flask import Flask, render_template, Response,request, jsonify, send_file
import cv2
from keras.models import model_from_json
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import csv
import os
import sqlite3

app = Flask(__name__)

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

prediction_label = None

def detect_emotion():
    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1028, 600))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Resize image and extract features
            resized_image = cv2.resize(roi_gray, (48, 48))
            img = extract_features(resized_image)
            
            # Predict emotion
            pred = model.predict(img)
            global prediction_label
            prediction_label = labels[pred.argmax()]
            
            # Draw rectangle and display emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert frame to bytes
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/get_emotion')                
def get_emotion():
    return prediction_label

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/write_emotion')
# def detected_feed():
#     return Response(detected_emotion(), mimetype='text/plain')

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Iot(db.Model):
    sno = db.Column(db.Integer(), primary_key= True)
    email = db.Column(db.String(), nullable= False)
    name = db.Column(db.String(), nullable= False)
    EmotionDetected = db.Column(db.String(), nullable= False)
    SongRecommended = db.Column(db.String(), nullable= False)


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

@app.route('/form',methods=['GET','POST'])
def form():
    if request.method == "POST":
        EmotionDetected = request.form['EmotionDetected']
        SongRecommended = request.form['SongRecommended']
        name = request.form['name']
        email = request.form['email']
        iot = Iot(email=email,EmotionDetected=EmotionDetected,SongRecommended=SongRecommended,name=name)
        db.session.add(iot)
        db.session.commit()
    alldata = Iot.query.all()
    return render_template('form.html',alldata=alldata)

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

    cursor.execute('''
   CREATE TABLE IF NOT EXISTS Login (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       fullname TEXT,
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

# @app.route("/get_recommendations")
# def get_recommendations():
#     [emotion,df1] = max_emotion_reccomendation()
#     return jsonify({"detected_emotion":emotion,"music_data":df1.to_dict(orient="records")if df1 is not None else None})

# music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
# global df1
# show_text=[0]
# df1 = pd.read_csv(music_dist[show_text[0]])
# df1 = df1[['Name','Album','Artist','Link','Image']]
# df1 = df1.head(15)

@app.route('/read_csv')
def read_csv():
    songs = []
    emotionfile = f"{prediction_label}.csv"
    with open(emotionfile, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            songs.append({
                'Name': row['Name'],
                'Album': row['Album'],
                'Artist': row['Artist'],
                'Link': row['Link'],
                'Image': row['Image']
            })
    return jsonify(songs)

@app.route('/play/<path:filename>')
def play(filename):
    audio_folder = 'audio'
    file_path = os.path.join(audio_folder, filename)
    return send_file(file_path, as_attachment=False)

@app.route('/songs')
def songs():
    return render_template('songs.html')

if __name__ == '__main__':
    # app.run(debug=True,port=5001)
    port = int(os.environ.get('PORT', 10000))  # Use environment variable if available
    app.run(host='0.0.0.0', port=port, debug=False)