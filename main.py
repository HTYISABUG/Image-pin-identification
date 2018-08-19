import os
import time
import glob
from collections import namedtuple
from pprint import pprint

import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

import make_data

_FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode',      'train', 'must be one of train/eval/iden')
tf.app.flags.DEFINE_string('image_path', '',        'image path')
tf.app.flags.DEFINE_string('label_path', '',        'label path')
tf.app.flags.DEFINE_string('log_dir',   'logs',     'directory for logging.')

tf.app.flags.DEFINE_integer('width',       250, 'width of image')
tf.app.flags.DEFINE_integer('height',      50,  'height of image')
tf.app.flags.DEFINE_integer('batch_size',  16,  'batch size')
tf.app.flags.DEFINE_integer('num_classes', 16,  'number of classes')
tf.app.flags.DEFINE_integer('hidden_dim',  256, 'dimention of hidden layer')
tf.app.flags.DEFINE_integer('epochs',      100, 'round of training')

tf.app.flags.DEFINE_float('lr',               0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1,  'initial accumulator value for Adagrad')

def main(unused_args):
    if len(unused_args) != 1: raise Exception('Problem with flags: %s' % unused_argv)

    if not os.path.exists(_FLAGS.log_dir):
        if _FLAGS.mode == 'train': os.makedirs(_FLAGS.log_dir)
        else: raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (_FLAGS.log_dir))

    images, labels = make_data.read_data(_FLAGS.image_path, _FLAGS.label_path)
    train_data, test_data = make_data.split_data(images, labels)

    if _FLAGS.mode == 'train':
        model = build_model()
        run_training(model, train_data)
    elif _FLAGS.mode == 'eval':
        model = keras.models.load_model(os.path.join(_FLAGS.log_dir, 'weight.best.hdf5'))
        run_evaluate(model, test_data)
    elif _FLAGS.mode == 'iden':
        model = keras.models.load_model(os.path.join(_FLAGS.log_dir, 'weight.best.hdf5'))
        run_identify()
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/iden")

def build_model():

    hdf5_path = os.path.join(_FLAGS.log_dir, 'weight.best.hdf5')

    if os.path.exists(hdf5_path):
        return keras.models.load_model(hdf5_path)

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(_FLAGS.height, _FLAGS.width, 1)))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(_FLAGS.hidden_dim, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(_FLAGS.num_classes, activation='softmax'))

    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def run_training(model, data):
    model.fit(x=data.images,
              y=data.labels,
              batch_size=_FLAGS.batch_size,
              epochs=_FLAGS.epochs,
              validation_split=0.1,
              callbacks=[keras.callbacks.TensorBoard(log_dir=_FLAGS.log_dir),
                         keras.callbacks.ModelCheckpoint(os.path.join(_FLAGS.log_dir, 'weight.best.hdf5'), save_best_only=True)])

def run_evaluate(model, data):
    predicts = model.predict(data.images)

    print('accuracy on test set: %f' % (np.mean(np.argmax(predicts, axis=1) == np.argmax(data.labels, axis=1))))

def run_identify(model):
    image_list = sorted(glob.glob(_FLAGS.image_path), key=lambda path: int(os.path.basename(path).rstrip('.png')))
    images = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in image_list]

    images = np.array(images) / 255
    images = np.expand_dims(images, axis=-1)

    predict = model.predict(images, batch_size=len(images))

if __name__ == '__main__':
    tf.app.run()
