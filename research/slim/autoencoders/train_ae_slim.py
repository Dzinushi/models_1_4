from __future__ import division, print_function, absolute_import

import tensorflow as tf
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim
from datasets import dataset_factory
from autoencoders import ae_factory
from collections import defaultdict
from time import time
# from autoencoders.optimizers.ae_sdc_1.sgd import GradientDescentOptimizerSDC1
from autoencoders.optimizers.ae_sdc_1.sgd_v2 import GradientDescentOptimizerSDC1
from autoencoders.optimizers.ae_sdc_1.gradient import CustomGradientSDC1 as gradient_custom_cpu
# from autoencoders.optimizers.ae_sdc_1.gradient_gpu import CustomGradientSDC1 as gradient_custom_cpu
from autoencoders.optimizers.optimizer_utils import Formulas
from datetime import datetime, timedelta
import numpy as np
import os
from autoencoders.dataset_np.db_np_flowers import DatabaseFlowers, Table

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

######################
# Optimization Flags #
######################

# tf.app.flags.DEFINE_float(
#     'weight_decay', 0.00004, 'The weight decay on the model weights.')
#
# tf.app.flags.DEFINE_string(
#     'optimizer', 'rmsprop',
#     'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
#     '"ftrl", "momentum", "sgd" or "rmsprop".')
#
# tf.app.flags.DEFINE_float(
#     'adadelta_rho', 0.95,
#     'The decay rate for adadelta.')
#
# tf.app.flags.DEFINE_float(
#     'adagrad_initial_accumulator_value', 0.1,
#     'Starting value for the AdaGrad accumulators.')
#
# tf.app.flags.DEFINE_float(
#     'adam_beta1', 0.9,
#     'The exponential decay rate for the 1st moment estimates.')
#
# tf.app.flags.DEFINE_float(
#     'adam_beta2', 0.999,
#     'The exponential decay rate for the 2nd moment estimates.')
#
# tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
#
# tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
#                           'The learning rate power.')
#
# tf.app.flags.DEFINE_float(
#     'ftrl_initial_accumulator_value', 0.1,
#     'Starting value for the FTRL accumulators.')
#
# tf.app.flags.DEFINE_float(
#     'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
#
# tf.app.flags.DEFINE_float(
#     'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
#
# tf.app.flags.DEFINE_float(
#     'momentum', 0.9,
#     'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
#
# tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
#
# tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

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
    list_assigned_vars = defaultdict(lambda: None)
    with tf.Graph().as_default():
        sess_load = tf.Session()
        with sess_load:
            saver_load = tf.train.import_meta_graph(model_path + '.meta')
            tf.logging.info('\n' + model_path + '.meta - OK')
            saver_load.restore(sess_load, tf.train.latest_checkpoint('/'.join(model_path.split('/')[:-1])))
            vars_load = sess_load.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in vars_load:
                list_assigned_vars[var._shared_name] = var.eval()
            tf.logging.info('Pretrained vars successfully loaded')
        return list_assigned_vars


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


activation_dic = {'relu': tf.nn.relu,
                  'leakyrelu': tf.nn.leaky_relu,
                  'sigmoid': tf.nn.sigmoid,
                  'tanh': tf.nn.tanh}


def main(_):
    logdir_fn = lambda block_number: FLAGS.train_dir + '/train_block_{}/'.format(block_number)

    sdc_num = 1
    full_train_time = 0

    # =================================================================================================
    # Select autoencoder model #
    # =================================================================================================

    autoencoder_fn, autoencoder_loss_map_fn = ae_factory.get_ae_fn(FLAGS.ae_name)
    block_count = autoencoder_fn.block_number

    # Test data. Can be removed
    # w_block = list(range(block_count))

    for train_block_number in range(block_count):
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            # =================================================================================================
            # Dataset settings #
            # =================================================================================================

            dataset = DatabaseFlowers(FLAGS.dataset_dir)
            table = Table.flowers_train

            # Set dataset using dataset factory
            # dataset = dataset_factory.get_dataset(
            #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

            # Get images and labels from dataset by auto batch queue
            # image_size = autoencoder_fn.default_image_size
            # batch_queue = load_batch(dataset, FLAGS.batch_size, image_size, image_size, is_training=True)
            # images, labels = batch_queue.dequeue()

            # calculate the number of batches per epoch
            batch_per_ep = dataset.num_samples['train'] // FLAGS.batch_size

            # images = np.random.rand(1,28,28,3)
            # images = np.array(images, dtype=np.float32)

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

            # Set x1 new weights as transposing x0 weights
            x0_weights = None
            list_assign_recovery_weights = None
            list_assign_pretrained_vars = None
            x1_weights = tf.placeholder(dtype=tf.float32)
            train_vars = tf.trainable_variables()

            for var in train_vars:
                if var.name.find('weights:0') != -1:
                    if var.name.find('recovery') != -1:
                        list_assigned_vars = defaultdict(lambda: None)
                        list_assigned_vars[var._shared_name] = x1_weights
                        list_assign_recovery_weights = init_fn(list_assigned_vars)
                    else:
                        x0_weights = var

            # Set pretrained variables from pretrained layers
            if train_block_number > 0:
                checkpoint_path = tf.train.latest_checkpoint(logdir_fn(train_block_number - 1))
                list_assigned_vars = load_vars_from_path(checkpoint_path)
                list_assign_pretrained_vars = init_fn(list_assigned_vars)

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
            loss_op = (loss_list[0] + loss_list[1]) / 2.0
            tf.losses.add_loss(loss_op)

            # =================================================================================================
            # Optimizer
            # =================================================================================================

            # Create custom gradient for SDC1
            grad = {}
            for var in tf.trainable_variables():
                grad[var.name] = tf.placeholder(tf.float32, shape=var.shape, name=var._shared_name + '_grad')

            optimizer = GradientDescentOptimizerSDC1(grad, FLAGS.learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

            train_op = slim.learning.create_train_op(loss_op, optimizer)

            # =================================================================================================
            # Summaries
            # =================================================================================================

            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            # Add summaries for end_points.
            for end_point in end_points:
                x = end_points[end_point]
                summaries.add(tf.summary.histogram('activations/' + end_point, x))
                summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                                tf.nn.zero_fraction(x)))

            # Add summaries for losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

            # Add summaries for variables.
            for variable in slim.get_model_variables():
                summaries.add(tf.summary.histogram(variable.op.name, variable))

            # Add total_loss to summary.
            # summaries.add(tf.summary.scalar('total_loss', loss_op))

            # Merge all summaries together.
            summary_op = tf.summary.merge(list(summaries), name='summary_op')

            if FLAGS.num_epoch == -1:
                number_of_steps = FLAGS.max_number_of_steps
            else:
                number_of_steps = batch_per_ep * FLAGS.num_epoch

            # =================================================================================================
            # USER PARAMS for custom gradient
            # =================================================================================================

            # If input_sdc_1 is recovered first input layer, it's shape = (NHWC),  else = (HWCN)
            activation_name = str.lower(loss_map[0]['output'].name.split('/')[-1].split(':')[0])
            if activation_name == 'maximum':
                activation_name = str.lower(loss_map[0]['output'].name.split('/')[2])

            global_step = tf.train.get_global_step()

            # =================================================================================================
            # Session
            # =================================================================================================

            saver = tf.train.Saver()
            checkpoint_path = logdir_fn(train_block_number) + 'model.ckpt'
            if not os.path.exists(logdir_fn(train_block_number)):
                os.makedirs(logdir_fn(train_block_number))

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

            # Create session
            session = tf.train.MonitoredTrainingSession(save_summaries_secs=100000000,
                                                        checkpoint_dir=logdir_fn(train_block_number),
                                                        config=config)

            # Save block N graph
            tf.train.write_graph(get_session(session).graph, logdir_fn(train_block_number), 'graph.pbtxt')

            with session._tf_sess() as sess:

                # Initialize recovery weights by transposed input weights
                if global_step.eval(session=sess) == 0:
                    # x_shape = x0_weights.shape.as_list()
                    x0_weights_np = x0_weights.eval(session=sess)
                    # x1_weights_np = np.zeros_like(x0_weights_np)

                    # for q in range(x_shape[2]):
                    #     for k in range(x_shape[3]):
                    #         x1_weights_np[:, :, q, k] = np.transpose(x0_weights_np[:, :, q, k])
                    if len(x0_weights_np.shape) != 2:
                        sess.run(list_assign_recovery_weights, feed_dict={x1_weights: x0_weights_np})
                    else:
                        sess.run(list_assign_recovery_weights, feed_dict={x1_weights: np.transpose(x0_weights_np)})

                # Initialize pretrained layers by pretrained variables
                if list_assign_pretrained_vars is not None:
                    sess.run(list_assign_pretrained_vars)
                    tf.logging.info('Autoencoder pretrained variables successfully recovered!')

                total_loss = 0.0
                i = 0
                epoch_num = int(global_step.eval(session=sess) / batch_per_ep) + 1

                next_index = lambda step: (step * FLAGS.batch_size) % dataset.num_samples['train'] + 1

                while global_step.eval(session=sess) < number_of_steps:

                    # # Get images for training
                    # TODO: Check index
                    time_start = time()
                    index = next_index(global_step.eval(session=sess))
                    images_batch = dataset.select_batch_img_by_index(table,
                                                                     index=index,
                                                                     batch_size=FLAGS.batch_size)
                    image_db_time = time() - time_start

                    # Calc custom gradient
                    x, y = sess.run([[loss_map[0]['input'], loss_map[0]['output']],
                                     [loss_map[1]['input'], loss_map[1]['output']]],
                                    feed_dict={images_placeholder: images_batch})

                    # Create custom gradient for autoencoder_sdc_1
                    grad_custom = gradient_custom_cpu(grads=grad,
                                                      x=x,
                                                      y=y,
                                                      stride=stride,
                                                      padding=pad,
                                                      formulas=Formulas[FLAGS.formulas],
                                                      activation_name=activation_name)

                    # Train autoencoder
                    feed_dict = grad_custom.run()
                    feed_dict[images_placeholder.name] = images_batch
                    loss = sess.run(train_op, feed_dict=feed_dict)
                    time_end = time()

                    # For summary INFO
                    if global_step.eval(session=sess) % batch_per_ep == 0:
                        total_loss = 0.0
                        epoch_num += 1

                    total_loss += loss

                    if (global_step.eval(session=sess) + 1) % FLAGS.log_every_n_steps == 0:
                        tf.logging.info(
                            'Block: {}, Image from db: {:.3f} sec/batch, Step: {} (epoch {}), loss: {:.5f}, total_loss: {:.5f} ({:.3f} sec/step)'.format(
                                train_block_number,
                                image_db_time,
                                global_step.eval(session=sess) + 1,
                                epoch_num,
                                loss,
                                total_loss / ((global_step.eval(session=sess)) % batch_per_ep) + 1,
                                time_end - time_start))

                    full_train_time += time_end - time_start
                    i += 1

                    # Saver by step
                    if FLAGS.save_every_step is not None \
                            and global_step.eval(session=sess) % FLAGS.save_every_step == 0:
                        saver.save(get_session(sess), save_path=checkpoint_path, global_step=global_step)

                    # Test data. Can be removed
                    # all_vars = tf.all_variables()
                    # w_vars = {}
                    # for var in all_vars:
                    #     if var.name.find('weights:0') != -1:
                    #         w_vars[var.name] = var.eval(session=sess)
                    # w_block[train_block_number] = w_vars

                saver.save(get_session(sess), save_path=checkpoint_path, global_step=number_of_steps)

        dataset.close()

    full_train_time = timedelta(seconds=full_train_time)
    full_train_time = datetime(1, 1, 1) + full_train_time
    if full_train_time.day - 1 != 0:
        tf.logging.info('Full train time: %dd %dh %dm %ds', full_train_time.day - 1, full_train_time.hour,
                        full_train_time.minute, full_train_time.second)
    elif full_train_time.hour != 0:
        tf.logging.info('Full train time: %dh %dm %ds', full_train_time.hour, full_train_time.minute,
                        full_train_time.second)
    elif full_train_time.minute != 0:
        tf.logging.info('Full train time: %dm %ds', full_train_time.minute, full_train_time.second)
    else:
        tf.logging.info('Full train time: %ds', full_train_time.second)


if __name__ == '__main__':
    tf.app.run()
