import os

from tensorflow import keras
import cv2
import numpy as np

def run_identify(hdf5_path, image_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    model = keras.models.load_model(hdf5_path)
    image = np.expand_dims(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[np.newaxis], axis=-1)

    predict = model.predict(image)

    return '0123456789ABCDEF'[np.argmax(predict)]
