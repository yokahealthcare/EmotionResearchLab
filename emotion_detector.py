import os

import cv2
import pandas as pd
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

        if len(self.result) > 0:
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

    def get_emotion_result_df(self):
        ef = self.result_emotion
        ef = pd.DataFrame(
            ef.items(), columns=["emotion", "score"]
        )
        ef = ef.sort_values(
            by=["score"], ascending=False
        ).reset_index(drop=True)

        return ef

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
            return "Unknown", "Unknown"


# HOW TO RUN
if __name__ == '__main__':
    from ultralytics import YOLO


    class YoloFaceDetector:
        def __init__(self, yolo_model):
            self.model = YOLO(yolo_model)

        def run(self, source):
            return self.model.predict(source)[0]


    # Settings
    yolo = YoloFaceDetector("asset/yolo/yolov8l-face.pt")
    emot = EmotionDetector()

    face_cascade = cv2.CascadeClassifier('testing/haarcascade_frontalface_default.xml')
    img = cv2.imread("asset/img/face#1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cropped = img[int(y):int(y + h), int(x):int(x + w)]
        emotion, dominant_emotion = emot.run(cropped)
        cv2.putText(cropped, f"{dominant_emotion}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img[int(y):int(y + h), int(x):int(x + w)] = cropped

    # cap = cv2.VideoCapture("asset/video/gfriend.mp4")
    # while True:
    #     has_frame, img = cap.read()
    #     if not has_frame:
    #         break
    #
    #     result = yolo.run(img)
    #     boxes = result.boxes.xyxy.clone().tolist()
    #     for box in boxes:
    #         x1, y1, x2, y2 = box
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    #         cropped = img[y1:y2, x1:x2]
    #
    #         # ---------------- THIS ONE
    #         emotion, dominant_emotion = emot.run(cropped)
    #         # ---------------- THIS ONE
    #
    #         cv2.putText(cropped, f"{dominant_emotion}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #
    #         img[y1:y2, x1:x2] = cropped
    #

    # Display the outputq
    cv2.imshow('img', img)
    # if cv2.waitKey(20) & 0xFF == ord("q"):  # press q to quit
    #     break

    # cap.release()
    # cv2.destroyAllWindow()
