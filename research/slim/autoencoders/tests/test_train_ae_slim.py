from __future__ import division, print_function, absolute_import

import tensorflow as tf
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim
from autoencoders import ae_factory
from collections import defaultdict
from autoencoders.optimizers.ae_sdc_1.sgd_v2 import GradientDescentOptimizerSDC1
from autoencoders.dataset_np.db_np_flowers import DatabaseFlowers, Table
import os
import numpy as np

tf.app.flags.DEFINE_string('ae_name', None,
                           'Autoencoder model name (See slim/autoencoders/)')

tf.app.flags.DEFINE_string('formulas', None,
                           'Formulas for calculation different custom gradient. Must be "golovko" or "hinton"')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_every_step', None,
    'The frequency with which the model is saved, in seconds.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer(
    'num_epoch', -1, 'The number of training epoch')

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'activation', 'relu', 'Activation function')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.5, 'Gpu reservation for model calculation')

tf.app.flags.DEFINE_integer(
    'image_test_count', 1, '')

FLAGS = tf.app.flags.FLAGS


def load_batch(dataset, batch_size, height=224, width=224, is_training=True):
    ######################
    # Set provider train #
    ######################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])

    image = inception_preprocessing.preprocess_image(image, height, width, is_training=True)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2)
    return batch_queue


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def init_fn(list_pretrained_vars):
    global_vars = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    list_assign_op = []
    for var in global_vars:
        if list_pretrained_vars[var._shared_name] is not None:
            list_assign_op.append(var.assign(list_pretrained_vars[var._shared_name]))
    return list_assign_op


def load_vars_from_path(model_path):
    with tf.Graph().as_default():
        list_pretrained_vars = defaultdict(lambda: None)
        sess_load = tf.Session()
        with sess_load:
            saver_load = tf.train.import_meta_graph(model_path + '.meta')
            tf.logging.info('\n' + model_path + '.meta - OK')
            saver_load.restore(sess_load, tf.train.latest_checkpoint('/'.join(model_path.split('/')[:-1])))
            vars_load = sess_load.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in vars_load:
                list_pretrained_vars[var._shared_name] = var.eval()
            tf.logging.info('Pretrained vars successfully loaded')
        return list_pretrained_vars


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


activation_dic = {'relu': tf.nn.relu,
                  'leakyrelu': tf.nn.leaky_relu,
                  'sigmoid': tf.nn.sigmoid,
                  'tanh': tf.nn.tanh}


def save_in_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write('# Shape: {}\n'.format(data.shape))
        if len(data.shape) == 4:
            for batch in range(data.shape[0]):
                for map in range(data.shape[3]):
                    file.write('# Batch: {}, map: {}\n'.format(batch, map))
                    np.savetxt(file, data[batch, :, :, map], fmt='%f')
                    file.write('\n')
        elif len(data.shape) == 2:
            np.savetxt(file_path, data, fmt='%f')
        else:
            raise ValueError('Error: array shape must be 2d or 4d, not %d' % len(data.shape))


def main(_):
    logdir_fn = lambda block_number: FLAGS.train_dir + '/train_block_{}/'.format(block_number)

    sdc_num = 1
    list_assign = None

    # =================================================================================================
    # Select autoencoder model #
    # =================================================================================================

    autoencoder_fn, autoencoder_loss_map_fn = ae_factory.get_ae_fn(FLAGS.ae_name)
    block_count = autoencoder_fn.block_number

    for train_block_number in range(block_count):
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            # =================================================================================================
            # Dataset settings #
            # =================================================================================================

            dataset = DatabaseFlowers(FLAGS.dataset_dir)
            table = Table.flowers_validate

            # calculate the number of batches per epoch
            batch_per_ep = dataset.num_samples['train'] // FLAGS.batch_size

            # Get model data
            images_size = autoencoder_fn.default_image_size
            images_shape = (FLAGS.batch_size, images_size, images_size, 3)
            images_placeholder = tf.placeholder(tf.float32, shape=images_shape, name='input_image')

            ae_outputs, end_points, pad, stride = autoencoder_fn(images_placeholder,
                                                                 train_block_num=train_block_number,
                                                                 sdc_num=sdc_num,
                                                                 activation_fn=activation_dic[FLAGS.activation])

            # =================================================================================================
            # Assign data from trained layers #
            # =================================================================================================

            if train_block_number > 0:
                checkpoint_path = tf.train.latest_checkpoint(logdir_fn(train_block_number - 1))
                list_pretrained_vars = load_vars_from_path(checkpoint_path)
                list_assign = init_fn(list_pretrained_vars)

            # =================================================================================================
            # LOSS
            # =================================================================================================

            # Get loss_map
            loss_map = autoencoder_loss_map_fn(end_points, train_block_num=train_block_number, sdc_num=sdc_num)

            assert len(loss_map) == 2

            # Calc as sum((x0 - x1)^2) + sum((y0-y1)^2) / batch_size / (image_size)^2
            loss_list = []
            for loss in loss_map:
                loss_list.append(tf.reduce_mean(tf.square(loss_map[loss]['input'] - loss_map[loss]['output'])))

            # loss_op = tf.divide((loss_list[0] + loss_list[1]), tf.constant(2.0) * FLAGS.batch_size * pow(image_size, 2))
            loss_op = loss_list[0] + loss_list[1]
            tf.losses.add_loss(loss_op)

            # =================================================================================================
            # Optimizer
            # =================================================================================================

            # Create custom gradient for SDC1
            grad = {}
            for var in tf.trainable_variables():
                grad[var.name] = tf.placeholder(tf.float32, shape=var.shape, name=var._shared_name + '_grad')

            optimizer = GradientDescentOptimizerSDC1(grad, FLAGS.learning_rate)

            train_op = slim.learning.create_train_op(loss_op, optimizer)

            global_step = tf.train.get_global_step()

            # =================================================================================================
            # Session
            # =================================================================================================

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

            # Create session
            session = tf.train.MonitoredTrainingSession(save_summaries_secs=100000000,
                                                        checkpoint_dir=logdir_fn(train_block_number),
                                                        config=config)

            with session._tf_sess() as sess:

                # File tests save folder
                file_test_path = logdir_fn(train_block_number) + 'test/' + str(global_step.eval(session=sess)) + '/'

                if not os.path.exists(file_test_path):
                    os.makedirs(file_test_path)

                # Initialize pretrained variables values
                if list_assign is not None:
                    sess.run(list_assign)
                    tf.logging.info('Autoencoder pretrained variables successfully recovered!')

                for i in range(FLAGS.image_test_count):

                    # Get images for training
                    images_batch = dataset.select_batch_by_index(table,
                                                                 index=i+1,
                                                                 batch_size=FLAGS.batch_size)

                    # Calc custom gradient
                    x, y = sess.run([[loss_map[0]['input'], loss_map[0]['output']],
                                     [loss_map[1]['input'], loss_map[1]['output']]],
                                    feed_dict={images_placeholder: images_batch})

                    # .../test/global_step/img_number_x0.txt
                    save_in_file(file_test_path + str(i) + '_x0.csv', x[0])
                    save_in_file(file_test_path + str(i) + '_x1.csv', x[1])
                    save_in_file(file_test_path + str(i) + '_y0.csv', y[0])
                    save_in_file(file_test_path + str(i) + '_y1.csv', y[1])

                dataset.close()


if __name__ == '__main__':
    tf.app.run()
