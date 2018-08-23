import os
import glob
import argparse
from pprint import pprint
from collections import namedtuple
import pickle

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

    labels_ = np.zeros((len(labels), 16))
    labels_[np.arange(len(labels)), labels] = 1

    return images, labels_

def split_data(images, labels):
    num_image = len(images)
    num_label = len(labels)

    assert num_image == num_label, 'The number of images doesn\'t the same as the number of labels'

    total_ratio = _TRAIN_RATIO + _TEST_RATIO

    split_images = np.array_split(images, total_ratio)
    split_labels = np.array_split(labels, total_ratio)

    train_data = {'images': np.vstack(split_images[:_TRAIN_RATIO]) if len(images) >= total_ratio else images,
                  'labels': np.vstack(split_labels[:_TRAIN_RATIO]) if len(labels) >= total_ratio else labels}
    test_data  = {'images': np.vstack(split_images[_TRAIN_RATIO:]) if len(images) >= total_ratio else images,
                  'labels': np.vstack(split_labels[_TRAIN_RATIO:]) if len(labels) >= total_ratio else labels}

    dataset = namedtuple('Dataset', ['images', 'labels'])

    return dataset(**train_data), dataset(**test_data)

def read_full_data(image_path, label_path):
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

            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

            for x in range(15, 135, 35):
                images.append(image[:, x:x+35])
                labels.append('0123456789ABCDEF'.index(label[x//35]))

    images = np.array(images, dtype=np.float32) / 255
    images = np.expand_dims(images, axis=-1)

    labels_ = np.zeros((len(labels), 16))
    labels_[np.arange(len(labels)), labels] = 1

    return images, labels_

def write2bin(images, labels, save_path):
    idx = 0

    while os.path.exists(os.path.join(save_path, '%d.pickle' % (idx))):
        idx += 1

    pickle.dump({'images': images, 'labels': labels}, open(os.path.join(save_path, '%d.pickle' % (idx)), 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, help='Image path. You can use wildcard to input multiple images.')
    parser.add_argument('--label_path', type=str, help='Path of label.csv.')
    parser.add_argument('--save_path', default='.', help='The folder that binary file will be saved')

    args = parser.parse_args()

    images, labels = read_full_data(args.image_path, args.label_path)
    write2bin(images, labels, args.save_path)
