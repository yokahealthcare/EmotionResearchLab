import cv2
import numpy as np
import onnxruntime as rt
from keras.preprocessing import image


class GenderDetector:
    genders = ["Female", "Male"]

    def __init__(self, onnx_model_path):
        self.model = rt.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider']
        )

        self.gender = None
        self.dominant_gender = None

    def reset(self):
        self.gender = None
        self.dominant_gender = None

    @staticmethod
    def preprocess(face):
        face = cv2.resize(face, (224, 224))

        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face /= 255

        return face

    def predict(self, face):
        gender = self.model.run(['activation_3'], {"input": face})[0][0]
        dominant_gender = self.genders[np.argmax(gender)]

        return gender, dominant_gender

    @staticmethod
    def postprocess(gender, dominant_gender):
        dict_of_gender = {
            "gender": {
                "female": gender[0],
                "male": gender[1]
            },
            "dominant_gender": dominant_gender
        }

        return dict_of_gender

    def detect(self, face):
        self.reset()

        face = self.preprocess(face)
        self.gender, self.dominant_gender = self.predict(face)

        return self.postprocess(self.gender, self.dominant_gender)
