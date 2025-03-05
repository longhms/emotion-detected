import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.keras")

emotion_labels = ['Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        resized_face = cv2.resize(face_roi, (48, 48))
        input_face = np.expand_dims(resized_face, axis=0)
        input_face = np.expand_dims(input_face, axis=-1)
        input_face = input_face / 255.0

        prediction = model.predict(input_face)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
