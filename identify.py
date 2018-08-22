import os

from tensorflow import keras
import cv2
import numpy as np

def run_identify(hdf5_path, image):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    model = keras.models.load_model(hdf5_path)

    predict = model.predict(image)

    return '0123456789ABCDEF'[np.argmax(predict)]
