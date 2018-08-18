from pprint import pprint

import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, hps):
        self.__hps = hps

    def build(self):
        hps = self.__hps

        self.__image_batch = tf.placeholder(tf.float32, [None, hps.height, hps.width, 1], name='image_batch')
        self.__label_batch = tf.placeholder(tf.int32, [None, hps.num_targets], name='label_batch')

        self.__global_step = tf.train.get_or_create_global_step()

        image_batch = self.__image_batch
        convs       = self.__build_conv(image_batch)
        hiddens     = self.__build_hidden(convs)
        logits      = self.__build_classifier(hiddens)
        loss        = self.__build_loss(logits)
        train_op    = self.__build_train_op(loss)

        self.__logits   = logits
        self.__loss     = loss
        self.__train_op = train_op
        self.__summary  = tf.summary.merge_all()

        self.sess_hooks = [tf.train.NanTensorHook(loss)]

    def __build_conv(self, images):
        hps = self.__hps

        with tf.variable_scope('conv'):
            conv0   = tf.layers.conv2d(images, 32, (3, 3), activation=tf.nn.relu, name='conv0')
            pool0   = tf.layers.max_pooling2d(conv0, (2, 2), (2, 2), name='pool0')

            conv1   = tf.layers.conv2d(pool0, 64, (3, 3), activation=tf.nn.relu, name='conv1')
            pool1   = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

            dropout = tf.layers.dropout(pool1, rate=0.25)
            output  = tf.layers.flatten(dropout)

            return output

    def __build_hidden(self, convs):
        hps = self.__hps

        with tf.variable_scope('hidden'):
            hidden = tf.layers.dense(convs, 256, activation=tf.nn.relu, name='hidden')
            output = tf.layers.dropout(hidden)

            return output

    def __build_classifier(self, hiddens):
        hps = self.__hps

        with tf.variable_scope('classifier'):
            outputs = [tf.layers.dense(hiddens, hps.num_classes, name='dense_%d' % (i)) for i in range(hps.num_targets)]

            return outputs

    def __build_loss(self, logits):
        hps = self.__hps

        with tf.variable_scope('loss'):
            losses = []

            for i in range(hps.num_targets):
                losses.append(tf.losses.sparse_softmax_cross_entropy(self.__label_batch[:, i], logits[i]))

            loss = tf.reduce_sum(losses)

            tf.summary.scalar('loss', loss)

            return loss

    def __build_train_op(self, loss):
        hps = self.__hps

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdagradOptimizer(hps.lr, initial_accumulator_value=hps.adagrad_init_acc)
            train_op = optimizer.minimize(loss, global_step=self.__global_step, name='train_op')

            return train_op

    def run_training(self, sess, batch):
        rets = {'loss': self.__loss,
                'train_op': self.__train_op,
                'global_step': self.__global_step,}
        feed_dict={self.__image_batch: batch.images,
                   self.__label_batch: batch.labels,}

        return sess.run(rets, feed_dict=feed_dict)

    def run_identify(self, sess, images):
        feed_dict={self.__image_batch: images,}

        logits = sess.run(self.__logits, feed_dict=feed_dict)

        def softmax(a):
            exp  = np.exp(a)
            prob = exp / np.sum(exp, axis=1)[:, np.newaxis]
            return prob

        probs  = [softmax(logit) for logit in logits]
        labels = np.array([np.argmax(prob, axis=1) for prob in probs]).T

        for label in labels:
            print(''.join(['0123456789ABCDEF'[c] for c in label]))
