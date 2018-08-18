import os
import time
from collections import namedtuple
from pprint import pprint

import tensorflow as tf
import numpy as np
import cv2

from model import Model
import util
from batcher import batch_generator

_FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode',      'training', 'must be one of training/evaluate/identify')
tf.app.flags.DEFINE_string('image_path', '',        'image path')
tf.app.flags.DEFINE_string('label_path', '',        'label path')
tf.app.flags.DEFINE_string('log_dir',   'logs',     'directory for logging.')

tf.app.flags.DEFINE_integer('width',       250, 'width of image')
tf.app.flags.DEFINE_integer('height',      50,  'height of image')
tf.app.flags.DEFINE_integer('batch_size',  16,  'batch size')
tf.app.flags.DEFINE_integer('num_classes', 16,  'number of classes')
tf.app.flags.DEFINE_integer('num_targets', 4,   'targets per image')

tf.app.flags.DEFINE_float('lr',               0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1,  'initial accumulator value for Adagrad')

def main(unused_args):
    if len(unused_args) != 1: raise Exception('Problem with flags: %s' % unused_argv)

    if not os.path.exists(_FLAGS.log_dir):
        if _FLAGS.mode == 'training': os.makedirs(_FLAGS.log_dir)
        else: raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (_FLAGS.log_dir))

    hps_name = ['width', 'height', 'batch_size', 'num_classes', 'num_targets',
                'lr', 'adagrad_init_acc']
    hps = {}

    for k, v in _FLAGS.__flags.items():
        if k in hps_name: hps[k] = v.value

    hps = namedtuple('HyperParams', hps.keys())(**hps)

    model = Model(hps)

    if _FLAGS.mode == 'training':
        run_training(model)
    elif _FLAGS.mode == 'evaluate':
        pass
    elif _FLAGS.mode == 'identify':
        run_identify(model)
    else:
        raise ValueError("The 'mode' flag must be one of training/evaluate/identify")

def run_training(model):

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

            generator = batch_generator(_FLAGS.image_path, _FLAGS.label_path, _FLAGS.batch_size)

            while True:
                tf.logging.info('running training step...')

                ts = time.time()

                batch = next(generator)

                result = model.run_training(sess, batch)

                tf.logging.info('seconds for training step: %.3f', time.time() - ts)
                tf.logging.info('loss: %f', result['loss'])

    except KeyboardInterrupt:
        tf.logging.info('Caught keyboard interrupt on worker. Stopping supervisor...')

def run_identify(model):

    if os.path.isfile(_FLAGS.image_path):
        images = cv2.imread(_FLAGS.image_path, cv2.IMREAD_GRAYSCALE)[np.newaxis]
    else:
        image_list = sorted(os.listdir(_FLAGS.image_path), key=lambda s: int(s.rstrip('.png')))
        images = [cv2.imread(os.path.join(_FLAGS.image_path, image_name), cv2.IMREAD_GRAYSCALE) for image_name in image_list]

    images = np.array(images) / 255
    images = np.expand_dims(images, axis=-1)

    with tf.device('/cpu:0'):
        model.build()

    saver = tf.train.Saver()

    with tf.Session(config=util.get_config()) as sess:
        util.load_ckpt(sess, saver, _FLAGS.log_dir)

        model.run_identify(sess, images)

if __name__ == '__main__':
    tf.app.run()
