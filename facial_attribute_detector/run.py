import cv2
from ultralytics import YOLO

from emotion_detector.emotion_detector import EmotionDetector
from age_detector.age_detector import AgeDetector
from gender_detection.gender_detector import GenderDetector


class YoloFaceDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, device=0)[0]


if __name__ == '__main__':
    yolo = YoloFaceDetector("yolo-face/yolov8m-face.pt")
    emotion_detector = EmotionDetector("emotion_detector/model/facial_expression_model_weights_48x48x1.onnx")
    age_detector = AgeDetector("age_detector/model/age_model_weights_224x224x3.onnx")
    gender_detector = GenderDetector("gender_detection/model/gender_model_weights_224x224x3.onnx")

    cap = cv2.VideoCapture("../asset/img/aerith#4.jpg")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        result = yolo.run(frame)
        boxes = result.boxes.xyxy.clone().tolist()
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))

            face = frame[y1:y2, x1:x2]

            # If RetinaFace being USED - just use this
            # Facial attribute analysis (emotion, age, and gender)
            predicted_emotion = emotion_detector.detect(face)
            predicted_age = age_detector.detect(face)
            predicted_gender = gender_detector.detect(face)
            predicted_merged = {**predicted_emotion, **predicted_age, **predicted_gender}

            print(predicted_merged)

            # Annotation
            what_we_see = ["dominant_emotion", 'age', 'dominant_gender']
            for idx, key in enumerate(what_we_see):
                cv2.putText(face, f"{key.split('_')[-1]} : {predicted_merged[key]}", (0, h - (20 * idx)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

            frame[y1:y2, x1:x2] = face

        cv2.imshow("webcam", frame)

        if cv2.waitKey(0) & 0xFF == ord("q"):  # press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()
