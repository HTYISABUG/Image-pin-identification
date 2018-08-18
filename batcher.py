import os
import re

import pandas as pd
import cv2
import numpy as np

def batch_generator(image_path, label_path, batch_size):
    image_list = [f for f in os.listdir(image_path) if re.match(r'[0-9]*\.png', f)]
    label_df   = pd.read_csv(label_path, index_col=0)

    images = []
    labels = []

    for no, label in label_df.iterrows():
        image_name = '%d.png' % (no)

        if image_name not in image_list:
            continue

        image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)

        label = np.array(['0123456789ABCDEF'.index(c) for c in label['result']])

        images.append(image)
        labels.append(label)

    images = np.array(images) / 255
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels)

    while True:
        samples = np.random.choice(len(images), size=batch_size, replace=False)

        yield Batch(images[samples], labels[samples])

class Batch(object):

    def __init__(self, images, labels):
        self.__images = images
        self.__labels = labels

    @property
    def images(self):
        return self.__images

    @property
    def labels(self):
        return self.__labels
