import os

import cv2
import pandas as pd
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class EmotionDetector:
    DeepFace.build_model(model_name="Emotion")
    logger.info("Emotion model is just built")

    def __init__(self):
        self.result = None
        self.can_detected = False

    def detect(self, cf):
        self.result = DeepFace.analyze(
            img_path=cf,
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
            actions=["emotion"]
        )

        # Check is it possible for emotion to be detected
        if len(self.result) <= 0:
            self.can_detected = False
        else:
            self.can_detected = True
            self.result = self.result[0]['emotion']

    def can_detect(self):
        return self.can_detected

    def get_emotion(self):
        return self.result

    def get_emotion_df(self):
        ef = self.result
        ef = pd.DataFrame(
            ef.items(), columns=["emotion", "score"]
        )
        ef = ef.sort_values(
            by=["score"], ascending=False
        ).reset_index(drop=True)

        return ef

    def get_dominant_emotion(self):
        max_emotion = max(self.result, key=self.result.get)
        return max_emotion


# HOW TO RUN
if __name__ == '__main__':
    emot = EmotionDetector()

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('asset/model/haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread('asset/img/aerith#3.jpg')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cropped = img[y:y + h, x:x + w]
        emot.detect(cropped)
        if emot.can_detect():
            emotion = emot.get_emotion()
            print(emotion)
            dominant_emotion = emot.get_dominant_emotion()
            print(dominant_emotion)

    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()
