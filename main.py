import os
import time

import cv2
import imutils
import pandas as pd
from deepface import DeepFace
from deepface.commons.logger import Logger
from ultralytics import YOLO

from emotion_detector import EmotionDetector


class YoloFaceDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source)[0]


if __name__ == '__main__':
    emot = EmotionDetector()
    yolo = YoloFaceDetector("asset/yolo/yolov8l-face.pt")

    # Emotion detector
    # Global variables
    text_color = (255, 255, 255)

    # Settings
    time_threshold = 1
    frame_threshold = 20

    # Visualization
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequentially 5 frames
    freezed_frame = 0
    tic = time.time()

    cap = cv2.VideoCapture("asset/video/gfriend.mp4")

    while True:
        has_frame, img = cap.read()
        if not has_frame:
            break

        start = time.perf_counter()
        # Resize
        if img.shape[0] > 720:
            img = imutils.resize(img, width=1280)

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        if not freeze:
            try:
                faces = []

                result = yolo.run(img)
                for face in result.boxes.xyxy:
                    x1, y1, x2, y2 = face
                    w = x2 - x1
                    h = y2 - y1
                    x, y, w, h = int(x1), int(y1), int(w), int(h)

                    faces.append((x, y, w, h))
            except:
                faces = []

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        for x, y, w, h in faces:
            face_detected = True
            if face_index == 0:
                face_included_frames += 1  # increase frame for a single face

            # Draw bounding box around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img, str(frame_threshold - face_included_frames), (int(x + w / 4), int(y + h / 1.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, )

            detected_face = img[y: y + h, x: x + w]  # crop detected face
            detected_faces.append((x, y, w, h))
            face_index = face_index + 1

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze:
            toc = time.time()
            if (toc - tic) < time_threshold:
                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        # draw rectangle to main image
                        cv2.rectangle(freeze_img, (x, y), (x + w, y + h), (255, 255, 255), 1)
                        # extract detected face
                        custom_face = base_img[y: y + h, x: x + w]

                        # facial attribute analysis
                        emotion, dominant_emotion = emot.run(custom_face)

                        if emot.is_face_detected():
                            emotion_df = emot.get_emotion_result_df()

                            emo, score = emotion_df.iloc[0]
                            cv2.putText(
                                freeze_img, emo, (x + w, y + h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                            )

                            # background of mood box

                            # transparency
                            # overlay = freeze_img.copy()
                            # opacity = 0.4
                            #
                            # if x + w + pivot_img_size < resolution_x:
                            #     # right
                            #     cv2.rectangle(
                            #         freeze_img, (x + w, y), (x + w + pivot_img_size, y + h), (64, 64, 64), cv2.FILLED
                            #     )
                            #     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)
                            #
                            # elif x - pivot_img_size > 0:
                            #     # left
                            #     cv2.rectangle(
                            #         freeze_img, (x - pivot_img_size, y), (x, y + h), (64, 64, 64), cv2.FILLED,
                            #     )
                            #     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)
                            #
                            # for index, instance in emotion_df.iterrows():
                            #     current_emotion = instance["emotion"]
                            #     emotion_label = f"{current_emotion} "
                            #     emotion_score = instance["score"] / 100
                            #
                            #     bar_x = 35  # this is the size if an emotion is 100%
                            #     bar_x = int(bar_x * emotion_score)
                            #
                            #     if x + w + pivot_img_size < resolution_x:
                            #
                            #         text_location_y = y + 20 + (index + 1) * 20
                            #         text_location_x = x + w
                            #
                            #         if text_location_y < y + h:
                            #             cv2.putText(
                            #                 freeze_img, emotion_label, (text_location_x, text_location_y),
                            #                 cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1
                            #             )
                            #
                            #             cv2.rectangle(
                            #                 freeze_img,
                            #                 (x + w + 70, y + 13 + (index + 1) * 20),
                            #                 (
                            #                     x + w + 70 + bar_x,
                            #                     y + 13 + (index + 1) * 20 + 5,
                            #                 ),
                            #                 (255, 255, 255),
                            #                 cv2.FILLED,
                            #             )
                            #
                            #     elif x - pivot_img_size > 0:
                            #
                            #         text_location_y = y + 20 + (index + 1) * 20
                            #         text_location_x = x - pivot_img_size
                            #
                            #         if text_location_y <= y + h:
                            #             cv2.putText(
                            #                 freeze_img, emotion_label, (text_location_x, text_location_y),
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1
                            #             )
                            #
                            #             cv2.rectangle(
                            #                 freeze_img,
                            #                 (
                            #                     x - pivot_img_size + 70,
                            #                     y + 13 + (index + 1) * 20,
                            #                 ),
                            #                 (
                            #                     x - pivot_img_size + 70 + bar_x,
                            #                     y + 13 + (index + 1) * 20 + 5,
                            #                 ),
                            #                 (255, 255, 255),
                            #                 cv2.FILLED,
                            #             )

                        tic = time.time()  # in this way, freezed image can show 5 seconds

                time_left = int(time_threshold - (toc - tic) + 1)

                cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
                cv2.putText(
                    freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 1
                )

                cv2.imshow("img", freeze_img)
                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0
        else:
            cv2.imshow("img", img)

        if cv2.waitKey(20) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
