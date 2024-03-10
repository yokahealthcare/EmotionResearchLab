import os

import cv2
import imutils
from ultralytics import YOLO

from deepface import DeepFace

# from deepface.models.FacialRecognition import FacialRecognition
# from deepface.commons.logger import Logger
#
# logger = Logger(module="commons.realtime")
#
# # dependency configuration
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# class YoloPersonDetector:
#     def __init__(self, yolo_model):
#         self.model = YOLO(yolo_model)
#
#     def run(self, source):
#         return self.model.track(source, stream=True, tracker="bytetrack.yaml", persist=True)

# class EmotionDetector:
#     # global variables
#     text_color = (255, 255, 255)
#     pivot_img_size = 112  # face recognition result image
#
#     enable_emotion = True
#     enable_age_gender = False
#
#     DeepFace.build_model(model_name="Age")
#     logger.info("Age model is just built")
#     DeepFace.build_model(model_name="Gender")
#     logger.info("Gender model is just built")
#     DeepFace.build_model(model_name="Emotion")
#     logger.info("Emotion model is just built")
#
#     def __init__(self):
#         self.model_name = "VGG-Face"
#         self.time_threshold = 5
#         self.frame_threshold = 5
#     def analyze(self, tuple_of_faces):
#         for x1, y1, x2, y2 in tuple_of_faces:


if __name__ == '__main__':
    for emotion in DeepFace.stream(source="../asset/video/gfriend.mp4", time_threshold=1, frame_threshold=10):
        print(emotion)

    # yolo = YoloPersonDetector("../asset/yolo/yolov8l-face.pt")
    #
    # for result in yolo.run("../asset/video/rollin720.mp4"):
    #     original_frame = result.orig_img
    #
    #     # Resize
    #     if original_frame.shape[0] > 720:
    #         print("Resized to 720")
    #         original_frame = imutils.resize(original_frame, width=1280)
    #
    #     raw_img = original_frame.copy()
    #     resolution_x = original_frame.shape[1]
    #     resolution_y = original_frame.shape[0]
    #
    #     boxes = result.boxes
    #     if len(boxes) > 0:
    #         for head in boxes:
    #             x1, y1, x2, y2 = head
    #             x1 = int(x1 * resolution_x)
    #             y1 = int(y1 * resolution_y)
    #             x2 = int(x2 * resolution_x)
    #             y2 = int(y2 * resolution_y)
    #
    #             # Draw bounding box of cropped head
    #             cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    #
    #             # Facial analysis attribute
    #     else:
    #         face_included_frames = 0
    #
    #     # Plot
    #     cv2.imshow("webcam", original_frame)
    #     # Wait for a key event and get the ASCII code
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
