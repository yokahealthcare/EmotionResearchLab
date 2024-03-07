import imutils
from deepface import DeepFace
from ultralytics import YOLO
import cv2
import torch


class EmotionDetector:
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn']

    def __init__(self):
        pass

    def detect(self):
        pass


class YoloPersonDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.track(source, classes=[0], stream=True, tracker="bytetrack.yaml", persist=True)


if __name__ == '__main__':
    yolo = YoloPersonDetector("yolov8x.pt")
    emot = EmotionDetector()
    for result in yolo.run("asset/video/rollin1080.mp4"):
        original_frame = result.orig_img
        result_frame = result.plot()

        # Resize
        frame_height = result_frame.shape[0]
        frame_width = result_frame.shape[1]
        if frame_height > 720:
            print("Resized to 720")
            result_frame = imutils.resize(result_frame, width=1280)
            frame_height = result_frame.shape[0]
            frame_width = result_frame.shape[1]

        boxes = result.boxes
        # Cut the head
        head_xyxyns = boxes.xyxyn.clone()    # Separate shared memory from orginal
        for head in head_xyxyns:
            head[-1] *= 0.7     # Define how big the height of cropped head

            head[0] *= frame_width
            head[2] *= frame_width
            head[1] *= frame_height
            head[3] *= frame_height
            head = head.int().tolist()
            cv2.rectangle(result_frame, (head[0], head[1]), (head[2], head[3]), (0, 255, 0), 2)

        # Plot
        cv2.imshow("webcam", result_frame)

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
