import cv2
import tensorflow as tf
from tensorflow import keras
from keras import models
from tensorflow.keras.models import load_model 
import numpy as np
print(tf.__version__)
print(keras.__version__)
model = load_model('mlproj.h5') 

emotion_labels = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's face detection cascade to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the detected face region
        face_roi = gray[y:y + h, x:x + w]
      
        resized_face = cv2.resize(face_roi, (48, 48))
        
        # Normalize the pixel values to range [0, 1]
        normalized_face = resized_face / 255.0
    
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
     
        predictions = model.predict(reshaped_face)
        print(f'Raw predictions: {predictions}')
        emotion_label = emotion_labels[np.argmax(predictions)]
        
        # Overlay the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
