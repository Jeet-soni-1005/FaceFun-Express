from flask import Flask, render_template, Response
import cv2
import cvlib as cv
import joblib
import numpy as np

# Load the trained model
loaded_model = joblib.load('emotion_detection_model.pkl')

# Load overlay images (goggles and Santa hat)
goggles = cv2.imread('goggles.png', cv2.IMREAD_UNCHANGED)
cap = cv2.imread('cap.png', cv2.IMREAD_UNCHANGED)

def add_overlay(frame, overlay, position, size):
    x, y = position
    width, height = int(size[0]), int(size[1])
    overlay = cv2.resize(overlay, (width, height))

    h, w, _ = overlay.shape
    roi = frame[y:y + h, x:x + w]

    if roi.shape[0] != h or roi.shape[1] != w:
        return
 
    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        roi[:, :, c] = (overlay_alpha * overlay_rgb[:, :, c] +
                        (1 - overlay_alpha) * roi[:, :, c])

    frame[y:y + h, x:x + w] = roi

# Initialize Flask app
app = Flask(__name__)

def generate_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces, confidences = cv.detect_face(frame)

        for face, confidence in zip(faces, confidences):
            try:
                (start_x, start_y, end_x, end_y) = face
                face_width = end_x - start_x
                face_height = end_y - start_y

                face_crop = frame[start_y:end_y, start_x:end_x]
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resize = cv2.resize(face_gray, (48, 48))
                face_flat = face_resize.flatten()

                emotion = loaded_model.predict([face_flat])[0]

                if emotion == 3:
                    goggles_width = int(face_width)
                    goggles_height = int(goggles_width * goggles.shape[0] / goggles.shape[1])
                    goggles_x = start_x
                    goggles_y = start_y + int(face_height * 0.24)
                    add_overlay(frame, goggles, (goggles_x, goggles_y), (goggles_width, goggles_height))

                    hat_width = int(face_width * 1.5)
                    hat_height = int(hat_width * cap.shape[0] / cap.shape[1])
                    hat_x = start_x - 15
                    hat_y = start_y - int(hat_height * 0.8)
                    add_overlay(frame, cap, (hat_x, hat_y), (hat_width, hat_height))

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                label = f'Emotion: {emotion}'
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
