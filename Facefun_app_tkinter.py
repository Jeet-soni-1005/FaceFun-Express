import cv2
import cvlib as cv
import joblib
import numpy as np
import threading
from tkinter import Tk, Button, Label

# Load the trained model
loaded_model = joblib.load('emotion_detection_model.pkl')

# Load overlay images (goggles and Santa hat)
goggles = cv2.imread('goggles.png', cv2.IMREAD_UNCHANGED)
cap = cv2.imread('cap.png', cv2.IMREAD_UNCHANGED)

# Global variables for controlling the video feed
running = False
cap = None

# Function to add an overlay to the frame
def add_overlay(frame, overlay, position, size):
    x, y = position
    width, height = int(size[0]), int(size[1])
    overlay = cv2.resize(overlay, (width, height))

    h, w, _ = overlay.shape
    roi = frame[y:y + h, x:x + w]

    # Ensure ROI is within the frame boundaries
    if roi.shape[0] != h or roi.shape[1] != w:
        return

    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        roi[:, :, c] = (overlay_alpha * overlay_rgb[:, :, c] +
                        (1 - overlay_alpha) * roi[:, :, c])

    frame[y:y + h, x:x + w] = roi

# Function to process the webcam feed
def emotion_detection():
    global running, cap

    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces, confidences = cv.detect_face(frame)

        for face, confidence in zip(faces, confidences):
            try:
                (start_x, start_y, end_x, end_y) = face
                face_width = end_x - start_x
                face_height = end_y - start_y

                # Crop and preprocess the face for emotion detection
                face_crop = frame[start_y:end_y, start_x:end_x]
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resize = cv2.resize(face_gray, (48, 48))
                face_flat = face_resize.flatten()

                # Perform emotion prediction
                emotion = loaded_model.predict([face_flat])[0]

                # Overlay goggles and Santa hat if emotion is "Happy" (emotion == 3)
                if emotion == 3:
                    # Goggles Placement
                    goggles_width = int(face_width)
                    goggles_height = int(goggles_width * goggles.shape[0] / goggles.shape[1])  # Maintain aspect ratio
                    goggles_x = start_x
                    goggles_y = start_y + int(face_height * 0.24)
                    add_overlay(frame, goggles, (goggles_x, goggles_y), (goggles_width, goggles_height))

                    # Santa Hat Placement
                    hat_width = int((face_width) * 1.45)
                    hat_height = int((hat_width * cap.shape[0] / cap.shape[1]))  # Maintain aspect ratio
                    hat_x = int(start_x-50)
                    hat_y = int(start_y - hat_height*0.6)  # Slight overlap with forehead
                    add_overlay(frame, cap, (hat_x, hat_y), (hat_width, hat_height))

                # Draw bounding box around the face
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

                # Emotion labels mapping
                emotion_labels = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

                # Display the predicted emotion
                label = f'Emotion: {emotion_labels.get(emotion, "unknown")}'
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


            except Exception as e:
                print(f"Error processing face: {e}")

        # Display the resulting frame
        cv2.imshow('Emotion Detection with Overlays', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start the detection
def start_detection():
    global running
    if not running:
        running = True
        threading.Thread(target=emotion_detection).start()

# Function to stop the detection
def stop_detection():
    global running, cap
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    app.destroy()

# Create the Tkinter UI
app = Tk()
app.title("Emotion Detection App")
app.geometry("300x200")

label = Label(app, text="Emotion Detection with Overlays", font=("Arial", 14))
label.pack(pady=20)

start_button = Button(app, text="Start Detection", command=start_detection, font=("Arial", 12), bg="green", fg="white")
start_button.pack(pady=10)

stop_button = Button(app, text="Stop Detection", command=stop_detection, font=("Arial", 12), bg="red", fg="white")
stop_button.pack(pady=10)

app.mainloop()
