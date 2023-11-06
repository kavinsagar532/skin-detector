from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

def get_camera():
    # Attempt to open the default camera (camera index 0) or an alternative (camera index 1)
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)
        
    if not camera.isOpened():
        raise Exception("Could not open the camera.")
    
    return camera

def detect_skin(camera):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    while True:
        grabbed, frame = camera.read()
        
        if not grabbed:
            break

        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        _, jpeg = cv2.imencode('.jpg', np.hstack([frame, skin]))
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    camera = get_camera()
    return Response(detect_skin(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
