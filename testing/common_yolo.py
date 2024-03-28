import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.predict(source, device=0)[0]


def main():
    yolo = YoloDetector("yolov8s.onnx")

    cap = cv2.VideoCapture("../asset/video/rollin720.mp4")
    while True:
        has_frame, img = cap.read()
        if not has_frame:
            break
        w, h = img.shape[1], img.shape[0]

        result = yolo.run(img)

        # Display the output
        cv2.imshow('img', result.plot())
        if cv2.waitKey(20) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
