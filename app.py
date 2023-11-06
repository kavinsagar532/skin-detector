from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Skin detection parameters
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

def detect_skin(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # Apply processing steps (erosion, dilation, etc.) to skinMask here
    # ...

    return skinMask

def webcam():
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        skin_mask = detect_skin(frame)
        frame_with_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
        frame_with_skin = cv2.cvtColor(frame_with_skin, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', frame_with_skin)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
