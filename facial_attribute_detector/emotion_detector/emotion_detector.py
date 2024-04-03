import cv2
import numpy as np
import onnxruntime as rt
from keras.preprocessing import image


class EmotionDetector:
    list_of_emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    def __init__(self, onnx_model_path):
        self.model = rt.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider']
        )

        self.emotion = None
        self.dominant_emotion = None

    def reset(self):
        self.emotion = None
        self.dominant_emotion = None

    @staticmethod
    def preprocess(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))

        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Make the size (1,48,48,1)

        face /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        return face

    def predict(self, face):
        emotion = self.model.run(['dense_3'], {"input": face})[0][0, :]
        dominant_emotion = self.list_of_emotions[np.argmax(emotion)]

        return emotion, dominant_emotion

    @staticmethod
    def postprocess(emotion, dominant_emotion):
        dict_of_emotion = {
            "emotion": {
                "angry": emotion[0],
                "disgust": emotion[1],
                "fear": emotion[2],
                "happy": emotion[3],
                "sad": emotion[4],
                "surprise": emotion[5],
                "neutral": emotion[6]
            },
            "dominant_emotion": dominant_emotion
        }

        return dict_of_emotion

    def detect(self, face):
        self.reset()

        face = self.preprocess(face)
        self.emotion, self.dominant_emotion = self.predict(face)

        return self.postprocess(self.emotion, self.dominant_emotion)
