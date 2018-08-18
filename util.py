import tensorflow as tf

def get_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config

def load_ckpt(sess, saver, ckpt_dir):

    while True:

        try:
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
            saver.restore(sess, ckpt_state.model_checkpoint_path)

            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

            return ckpt_state.model_checkpoint_path

        except:
            tf.logging.info('Failed to load checkpoint from %s. Sleeping for %i secs...', ckpt_dir, 10)

            time.sleep(10)
