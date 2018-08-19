import os
import glob
import argparse
from pprint import pprint
from collections import namedtuple

import cv2
import numpy as np

_TRAIN_RATIO = 8
_TEST_RATIO  = 2

def read_data(image_path, label_path):
    image_list = glob.glob(image_path)
    dir_name   = os.path.dirname(image_list[0])

    images = []
    labels = []

    with open(label_path, 'r') as fp:

        for line in fp:
            image_name, label = line.strip().split(',')
            image_name = os.path.join(dir_name, image_name + '.png')

            if image_name not in image_list:
                continue

            images.append(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))
            labels.append('0123456789ABCDEF'.index(label))

    images = np.array(images, dtype=np.float32) / 255
    images = np.expand_dims(images, axis=-1)
    labels = np.eye(len(labels), 16)[np.array(labels)]

    return images, labels

def split_data(images, labels):
    num_image = len(images)
    num_label = len(labels)

    assert num_image == num_label, 'The number of images doesn\'t the same as the number of labels'

    total_ratio = _TRAIN_RATIO + _TEST_RATIO

    split_images = np.array_split(images, total_ratio)
    split_labels = np.array_split(labels, total_ratio)

    train_data = {'images': np.vstack(split_images[:_TRAIN_RATIO]),
                  'labels': np.vstack(split_labels[:_TRAIN_RATIO])}
    test_data  = {'images': np.vstack(split_images[_TRAIN_RATIO:]),
                  'labels': np.vstack(split_labels[_TRAIN_RATIO:])}

    dataset = namedtuple('Dataset', ['images', 'labels'])

    return dataset(**train_data), dataset(**test_data)
