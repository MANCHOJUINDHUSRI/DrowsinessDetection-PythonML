import cv2
import numpy as np
import tensorflow as tf
from pygame import mixer

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained drowsiness detection model
model = tf.keras.models.load_model('drowsiness_model.h5')

# Define the labels for drowsiness and alertness
labels = ['Alert', 'Drowsy']

# Define the threshold for drowsiness detection
threshold = 0.5

# Initialize the alarm sound
mixer.init()
alarm_sound = mixer.Sound('alarm.mp3')

# Define variables for tracking eye closure
closed_eyes_count = 0
eyes_closed = False
eyes_opened = False

# Define the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]
        
        # Preprocess the face image
        face = cv2.resize(face, (145, 145))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.reshape(-1, 145, 145, 3)
        
        # Perform drowsiness prediction
        predictions = model.predict(face)
        drowsiness_prob = predictions[0][0]
        
        # Determine the label based on the predicted probability
        if drowsiness_prob >= threshold:
            label = labels[1]  # Drowsy
            closed_eyes_count += 1
        else:
            label = labels[0]  # Alert
            closed_eyes_count = 0
        
        # Display the label and probability on the frame
        cv2.putText(frame, f'{label}: {drowsiness_prob:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Check for eye closure
    if closed_eyes_count >= 8:
        if not eyes_closed:
            eyes_closed = True
            eyes_opened = False
            alarm_sound.play(-1)  # Play the alarm sound continuously
            print("Eyes Closed!")
    
    # Check for eye opening
    if closed_eyes_count ==0:
        if not eyes_opened:
            eyes_opened = True
            eyes_closed = False
            alarm_sound.stop()  # Stop the alarm sound
            print("Eyes Opened!")
    
    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, stop the alarm sound, and close the windows
cap.release()
alarm_sound.stop()
cv2.destroyAllWindows
