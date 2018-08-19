import os
import time
import glob
from collections import namedtuple
from pprint import pprint

import tensorflow as tf
import numpy as np
import cv2

from model import Model
import util
from batcher import batch_generator
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

tf.app.flags.DEFINE_float('lr',               0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1,  'initial accumulator value for Adagrad')

def main(unused_args):
    if len(unused_args) != 1: raise Exception('Problem with flags: %s' % unused_argv)

    if not os.path.exists(_FLAGS.log_dir):
        if _FLAGS.mode == 'train': os.makedirs(_FLAGS.log_dir)
        else: raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (_FLAGS.log_dir))

    hps_name = ['width', 'height', 'batch_size', 'num_classes', 'num_targets', 'hidden_dim',
                'lr', 'adagrad_init_acc']
    hps = {}

    for k, v in _FLAGS.__flags.items():
        if k in hps_name: hps[k] = v.value

    hps = namedtuple('HyperParams', hps.keys())(**hps)

    images, labels = make_data.read_data(_FLAGS.image_path, _FLAGS.label_path)
    train_data, val_data, test_data = make_data.split_data(images, labels)

    model = Model(hps)

    if _FLAGS.mode == 'train':
        run_training(model, train_data, val_data)
    elif _FLAGS.mode == 'eval':
        pass
    elif _FLAGS.mode == 'iden':
        run_identify(model)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/iden")

def run_training(model, train_data, val_data):

    tf.logging.set_verbosity(tf.logging.INFO)

    if not os.path.exists(_FLAGS.log_dir): os.makedirs(_FLAGS.log_dir)

    model.build()

    try:
        sess_params = {
            'checkpoint_dir': _FLAGS.log_dir,
            'config': util.get_config(),
            'hooks': model.sess_hooks
        }

        with tf.train.MonitoredTrainingSession(**sess_params) as sess:
            tf.logging.info('starting run_training')

            generator = batch_generator(train_data, _FLAGS.batch_size)

            while True:
                tf.logging.info('running training step...')

                ts = time.time()

                batch = next(generator)

                result = model.run_training(sess, batch)

                tf.logging.info('seconds for training step: %.3f', time.time() - ts)
                tf.logging.info('loss: %f', result['loss'])

                cross_val = model.run_identify(sess._tf_sess(), val_data.images)
                accuracy  = np.mean(cross_val == val_data.labels)

                tf.logging.info('cross validation accuracy: %f', accuracy)

    except KeyboardInterrupt:
        tf.logging.info('Caught keyboard interrupt on worker. Stopping supervisor...')

def run_identify(model):
    image_list = sorted(glob.glob(_FLAGS.image_path), key=lambda path: int(os.path.basename(path).rstrip('.png')))
    images = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in image_list]

    images = np.array(images) / 255
    images = np.expand_dims(images, axis=-1)

    with tf.device('/cpu:0'):
        model.build()

    saver = tf.train.Saver()

    with tf.Session(config=util.get_config()) as sess:
        util.load_ckpt(sess, saver, _FLAGS.log_dir)

        labels = model.run_identify(sess, images).T[0]

        for label in labels: print('0123456789ABCDEF'[label])

if __name__ == '__main__':
    tf.app.run()
