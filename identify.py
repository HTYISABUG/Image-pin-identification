import os

from tensorflow import keras
import cv2
import numpy as np

def run_identify(hdf5_path, image_list):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    model  = keras.models.load_model(hdf5_path)

    images = np.array(image_list, dtype=np.float32) / 255
    images = np.expand_dims(images, axis=-1)

    predicts = model.predict(images)

    return ''.join(['0123456789ABCDEF'[l] for l in np.argmax(predicts, axis=1)])
