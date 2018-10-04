# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from collections import defaultdict
import os
from autoencoders.dataset_np.db_np_flowers import DatabaseFlowers, Table
from time import time
from datetime import datetime, timedelta


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

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

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'activation', 'relu', 'Activation function. May be "relu", "leakyrelu", "sigmoid" or "tanh".')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('num_epoch', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_string(
    'ae_path', None,
    'Path to autoencoder pretrained vars.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


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


def _get_init_fn(variables_to_restore):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.ae_path is not None:
        global_vars = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vars_load = {}
        for var in global_vars:
            if variables_to_restore[var._shared_name] is not None:
                vars_load[var._shared_name] = variables_to_restore[var._shared_name]
        tf.logging.info('\nAutoencoder pretrained variables successfully recovered')
        return slim.assign_from_values_fn(vars_load)

    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def load_vars_from_path(model_path, list_pretrained_vars):
    with tf.Graph().as_default():
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        sess_load = tf.Session()
        with sess_load:
            saver_load = tf.train.import_meta_graph(latest_checkpoint + '.meta')
            tf.logging.info('\n' + model_path + '.meta - OK')
            # saver_load.restore(sess_load, tf.train.latest_checkpoint('/'.join(model_path.split('/')[:-1])))
            saver_load.restore(sess_load, tf.train.latest_checkpoint(model_path))
            vars_load = sess_load.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in vars_load:
                list_pretrained_vars[var._shared_name] = var.eval()
            tf.logging.info('Pretrained vars from {} loaded'.format(model_path))
        return list_pretrained_vars


def load_vars_from_folder(folder_path):
    list_trainable_blocks = os.listdir(folder_path)
    list_pretrained_vars = defaultdict(lambda: None)
    for trainable_block in list_trainable_blocks:
        list_pretrained_vars = load_vars_from_path(model_path=folder_path + '/' + trainable_block,
                                                   list_pretrained_vars=list_pretrained_vars)
    return list_pretrained_vars


activation_dic = {'relu': tf.nn.relu,
                  'leakyrelu': tf.nn.leaky_relu,
                  'sigmoid': tf.nn.sigmoid,
                  'tanh': tf.nn.tanh}


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    variables_to_restore = defaultdict(lambda: None)
    if FLAGS.ae_path is not None:
        variables_to_restore = load_vars_from_folder(FLAGS.ae_path)
    # elif FLAGS.frozen_model_path is not None:
    #     variables_to_restore = load_vars_from_path(FLAGS.frozen_model_path)

    with tf.Graph().as_default():
        tf.set_random_seed(777)
        # tf.logging.info('Graph seed new: ', tf.get_default_graph().seed)

        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        dataset = DatabaseFlowers(FLAGS.dataset_dir)
        table = Table.flowers_train

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            activation=activation_dic[FLAGS.activation],
            is_training=True)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # Create image and label placeholders
        images_size = network_fn.default_image_size
        images_shape = (FLAGS.batch_size, images_size, images_size, 3)
        images_placeholder = tf.placeholder(tf.float32, shape=images_shape, name='image')
        labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, dataset.num_classes), name='label')

        logits, end_points = network_fn(images_placeholder)

        #############################
        # Specify the loss function #
        #############################
        if 'AuxLogits' in end_points:
            slim.losses.softmax_cross_entropy(
                end_points['AuxLogits'], labels_placeholder,
                label_smoothing=FLAGS.label_smoothing, weights=0.4,
                scope='aux_loss')
        slim.losses.softmax_cross_entropy(
            logits, labels_placeholder, label_smoothing=FLAGS.label_smoothing, weights=1.0)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        # first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

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

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(dataset.num_samples['train'], global_step)
            optimizer = _configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        # variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        # total_loss, clones_gradients = model_deploy.optimize_clones(
        #     clones,
        #     optimizer,
        #     var_list=variables_to_train)
        # Add total_loss to summary.

        summaries.add(tf.summary.scalar('total_loss', tf.losses.get_total_loss()))

        # Create gradient updates.
        train_op = slim.learning.create_train_op(tf.losses.get_total_loss(), optimizer)

        # grad_updates = optimizer.apply_gradients(clones_gradients,
        #                                          global_step=global_step)
        # update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        # with tf.control_dependencies([update_op]):
        #     train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        if FLAGS.num_epoch is not None:
            num_of_steps = int(FLAGS.num_epoch * dataset.num_samples['train'] / FLAGS.batch_size)
        else:
            num_of_steps = FLAGS.max_number_of_steps

        batch_per_ep = dataset.num_samples['train'] / FLAGS.batch_size

        ###########################
        # Kicks off the training. #
        ###########################

        saver = tf.train.Saver()

        # Create session
        session = tf.train.MonitoredTrainingSession(save_summaries_secs=100000000,
                                                    checkpoint_dir=FLAGS.train_dir)

        # Save block N graph
        tf.train.write_graph(get_session(session).graph, FLAGS.train_dir, 'graph.pbtxt')

        with session._tf_sess() as sess:

            total_loss = 0.0
            i = 0
            epoch_num = int(global_step.eval(session=sess) / batch_per_ep) + 1
            full_train_time = 0

            next_index = lambda step: (step * FLAGS.batch_size) % dataset.num_samples['train'] + 1

            while global_step.eval(session=sess) < num_of_steps:

                # # Get images for training
                # TODO: Check index
                time_start = time()
                index = next_index(global_step.eval(session=sess))
                images_batch, labels_batch = dataset.select_batch_by_index(table,
                                                                           index=index,
                                                                           batch_size=FLAGS.batch_size)
                db_time = time() - time_start

                # Train autoencoder
                loss = sess.run(train_op, feed_dict={images_placeholder: images_batch,
                                                     labels_placeholder: labels_batch})
                time_end = time()

                # For summary INFO
                if global_step.eval(session=sess) % batch_per_ep == 0:
                    total_loss = 0.0
                    epoch_num += 1

                total_loss += loss

                if (global_step.eval(session=sess) + 1) % FLAGS.log_every_n_steps == 0:
                    tf.logging.info(
                        'Data from db: {:.3f} sec/batch, Step: {} (epoch {}), loss: {:.5f}, total_loss: {:.5f} ({:.3f} sec/step)'.format(
                            db_time,
                            global_step.eval(session=sess) + 1,
                            epoch_num,
                            loss,
                            total_loss / ((global_step.eval(session=sess)) % batch_per_ep) + 1,
                            time_end - time_start))

                full_train_time += time_end - time_start
                i += 1

                # Saver by step
                # if FLAGS.save_every_step is not None \
                #         and global_step.eval(session=sess) % FLAGS.save_every_step == 0:
                #     saver.save(get_session(sess), save_path=FLAGS.train_dir, global_step=global_step)

            saver.save(get_session(sess), save_path=FLAGS.train_dir + 'model.ckpt', global_step=global_step)

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
