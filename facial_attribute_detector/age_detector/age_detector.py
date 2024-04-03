import cv2
import numpy as np
import onnxruntime as rt
from keras.preprocessing import image


class AgeDetector:
    def __init__(self, onnx_model_path):
        self.model = rt.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider']
        )

        self.age = None

    def reset(self):
        self.age = None

    @staticmethod
    def preprocess(face):
        face = cv2.resize(face, (224, 224))

        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face /= 255

        return face

    def predict(self, face):
        age = self.model.run(['activation_1'], {"input": face})[0][0]
        return age

    @staticmethod
    def postprocess(age_predictions):
        output_indexes = np.array(list(range(0, 101)))

        apparent_age = np.sum(age_predictions * output_indexes)

        result = {
            "age": int(apparent_age)
        }

        return result

    def detect(self, face):
        self.reset()

        face = self.preprocess(face)
        self.age = self.predict(face)

        return self.postprocess(self.age)
