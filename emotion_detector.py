import os

import cv2
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger(module="commons.realtime")
# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Face detection backends that DeepFace support
backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn',
]


class EmotionDetector:
    DeepFace.build_model(model_name="Emotion")
    logger.info("Emotion model is just built")

    def __init__(self):
        self.result = None
        self.result_xyxy = None
        self.result_emotion = None
        self.face_detected = False

    def reset(self):
        self.result = None
        self.result_xyxy = None
        self.result_emotion = None
        self.face_detected = False

    def detect(self, cf):
        self.reset()
        self.result = DeepFace.analyze(
            img_path=cf,
            detector_backend=backends[1],  # Doing face detection
            enforce_detection=False,
            silent=True,
            actions=["emotion"]
        )

        # Check if there is face detected
        if self.result[0]["face_confidence"] == 0:  # DeepFace automatically use whole image if no face detected
            self.face_detected = False
        else:
            self.face_detected = True
            self.result_emotion = self.result[0]["emotion"]

            x = self.result[0]["region"]["x"]
            y = self.result[0]["region"]["y"]
            w = self.result[0]["region"]["w"]
            h = self.result[0]["region"]["h"]
            self.result_xyxy = [
                x,
                y,
                x + w,
                y + h
            ]

    def is_face_detected(self):
        return self.face_detected

    def get_raw_result(self):
        return self.result

    def get_emotion_result(self):
        return self.result_emotion

    def get_xyxy_result(self):
        return self.result_xyxy

    def get_dominant_emotion(self):
        most_high_emotion = max(self.result_emotion, key=self.result_emotion.get)
        return most_high_emotion

    def run(self, cropped_face_region):
        self.detect(cropped_face_region)
        if self.is_face_detected():
            """
                self.result_xyxy            : Return list [x1, y1, x2, y2] where the face located
                self.get_emotion_result()   : Return dictionary | type of emotion (as key) & score (as value)
                self.get_dominant_emotion() : Return string of predicted emotion (the highest score)
            """
            return self.get_emotion_result(), self.get_dominant_emotion()

        else:
            return "No emotion detected", "No emotion detected"


# HOW TO RUN
if __name__ == '__main__':
    emot = EmotionDetector()

    # Read the input image
    img = cv2.imread('asset/img/aerith#5.jpg')

    emotion, dominant_emotion = emot.run(img)
    cv2.putText(img, f"{dominant_emotion}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()
