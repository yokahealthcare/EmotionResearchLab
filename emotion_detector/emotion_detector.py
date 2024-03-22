import cv2
import numpy as np
import onnxruntime as rt
import pandas as pd
from keras.preprocessing import image
from ultralytics import YOLO


class YoloFaceDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, device=0)[0]


class EmotionDetector:
    # List emotions on DeepFace
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    # ONNX execution engine
    providers = ['CUDAExecutionProvider']

    def __init__(self):
        self.face_detector = YoloFaceDetector("yolov8n-face.pt")

        self.result = []
        self.result_xyxy = None
        self.result_emotion = None
        self.face_detected = False

    def reset(self):
        self.result = []
        self.result_xyxy = None
        self.result_emotion = None
        self.face_detected = False

    def detect_faces(self, cf):
        faces = []

        result = self.face_detector.run(cf)
        xyxy = result.boxes.xyxy.clone().tolist()
        for x1, y1, x2, y2 in xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            faces.append((x, y, w, h))

        return faces

    def detect(self, cf):
        self.reset()
        for (x, y, w, h) in self.detect_faces(cf):
            detected_face = cf[y:y + h, x:x + w]
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48))

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            """
                img_pixels.shape = (1,48,48,1)
            """

            m = rt.InferenceSession("facial_expression_model_weights_48x48x1.onnx",
                                    providers=self.providers)
            onnx_pred = m.run(['dense_3'], {"input": img_pixels})[0][0, :]

            emotion = onnx_pred
            # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            dominant_emotion = self.emotions[np.argmax(onnx_pred)]

            self.result.append({
                "emotion": {
                    "angry": emotion[0],
                    "disgust": emotion[1],
                    "fear": emotion[2],
                    "happy": emotion[3],
                    "sad": emotion[4],
                    "surprise": emotion[5],
                    "neutral": emotion[6]
                },
                "dominant_emotion": dominant_emotion,
                "region": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                }
            })

        if len(self.result) > 0:
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
    # Settings
    yolo = YoloFaceDetector("yolov8n-face.pt")
    emot = EmotionDetector()

    cap = cv2.VideoCapture("sample.mp4")
    while True:
        has_frame, img = cap.read()
        if not has_frame:
            break
        w, h = img.shape[1], img.shape[0]

        result = yolo.run(img)
        boxes = result.boxes.xyxy.clone().tolist()
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cropped = img[y1:y2, x1:x2]

            # ---------------- THIS ONE
            emotion, dominant_emotion = emot.run(cropped)
            # ---------------- THIS ONE

            cv2.putText(cropped, f"{dominant_emotion}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            img[y1:y2, x1:x2] = cropped

        # Display the output
        cv2.imshow('img', img)
        if cv2.waitKey(20) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()