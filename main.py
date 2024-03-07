import time
from datetime import timedelta

import imutils
from deepface import DeepFace
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from keras.preprocessing import image


class EmotionDetector:
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn']

    def __init__(self):
        self.result = None

    def preprocessing(self, frame):
        frame = cv2.resize(frame, None, fx=0.7, fy=0.7).astype(np.float32)
        frame /= 255.0  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        return frame

    def run(self, source):
        self.result = DeepFace.analyze(img_path=source, actions=['emotion'], enforce_detection=False)

    def get_emotion(self):
        return self.result[0]["dominant_emotion"]

    def get_emotions_percentage(self):
        return self.result[0]["emotion"]


class YoloPersonDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.track(source, classes=[0], stream=True, tracker="bytetrack.yaml", persist=True)


if __name__ == '__main__':
    yolo = YoloPersonDetector("asset/yolo/yolov8m.pt")
    emot = EmotionDetector()

    for result in yolo.run("asset/video/rollin720.mp4"):
        start = time.perf_counter()

        original_frame = result.orig_img
        # Resize
        if original_frame.shape[0] > 720:
            print("Resized to 720")
            original_frame = imutils.resize(original_frame, width=1280)
        frame_height = original_frame.shape[0]
        frame_width = original_frame.shape[1]

        boxes = result.boxes
        # Cut the head area
        head_xyxyns = boxes.xyxyn.clone()    # Separate shared memory from orginal
        for head in head_xyxyns:
            x1, y1, x2, y2 = head
            y2 *= 0.7     # Define how big the height of cropped head

            x1 = int(x1 * frame_width)
            y1 = int(y1 * frame_height)
            x2 = int(x2 * frame_width)
            y2 = int(y2 * frame_height)

            # Draw bounding box of cropped head
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Emotion Detection
            cropped_head = original_frame[y1:y2, x1:x2]
            cropped_head = emot.preprocessing(cropped_head)
            emot.run(cropped_head)
            emotion = emot.get_emotion()

            cv2.putText(original_frame, emotion, (x1+10, y2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        end = time.perf_counter()
        print(f"Inference time for facial attribute analysis took {timedelta(seconds=(end - start))} \n")
        # Plot
        cv2.imshow("webcam", original_frame)

        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# # Load a model
# model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
# model.predict()
#
# pic1 = "asset/img/aerith#1.jpg"
# pic2 = "asset/img/aerith#2.jpg"

# demographies = DeepFace.analyze(img_path=pic1, detector_backend=backends[3])
# print(demographies)
#
# objs = DeepFace.analyze(img_path=pic1, actions=['age', 'gender', 'race', 'emotion'])
# print(objs)

"""

1. ATTRIBUTE ANALYSIS
2. CONVERT KE TENSORRT or ONNX
3. 
"""
