import cv2
import numpy as np
from keras.preprocessing import image
import onnxruntime as rt

providers = ['CUDAExecutionProvider']
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("../asset/img/face#1.jpg")
while True:
    has_frame, img = cap.read()
    if not has_frame:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        print(detected_face.shape)
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        img_pixels = np.squeeze(img_pixels, axis=3)

        m = rt.InferenceSession("../asset/model/facial_expression_model_weights_48x48.onnx", providers=providers)
        onnx_pred = m.run(['dense_3'], {"input": img_pixels})[0][0, :]

        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(onnx_pred)
        emotion = emotions[max_index]

        print(f"EMOTION : {emotion}")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

    cv2.imshow("webcam", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):  # press q to quit
        break
cap.release()
cv2.detriyAllWindows()
