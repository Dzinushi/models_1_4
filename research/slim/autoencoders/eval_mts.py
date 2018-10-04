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
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

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
    'batch_size', 50, 'The number of samples in each batch.')

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

    with tf.Graph().as_default():
        tf.set_random_seed(777)

        dataset = DatabaseFlowers(FLAGS.dataset_dir)
        table = Table.flowers_validate

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            activation=activation_dic[FLAGS.activation],
            is_training=False)

        # Create image and label placeholders
        images_size = network_fn.default_image_size
        images_shape = (FLAGS.batch_size, images_size, images_size, 3)
        images_placeholder = tf.placeholder(tf.float32, shape=images_shape, name='image')

        logits, _ = network_fn(images_placeholder)
        predictions = tf.argmax(logits, 1)
        # labels = tf.squeeze(labels_placeholder)
        #
        # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        #     'Recall_5': slim.metrics.streaming_recall_at_k(
        #         logits, labels, 5),
        # })
        #
        # # Print the summaries to screen.
        # for name, value in names_to_values.items():
        #     summary_name = 'eval/%s' % name
        #     op = tf.summary.scalar(summary_name, value, collections=[])
        #     op = tf.Print(op, [value], summary_name)
        #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        ###########################
        # Kicks off the training. #
        ###########################

        global_step = tf.train.get_or_create_global_step()

        # Create session
        session = tf.train.MonitoredTrainingSession(save_summaries_secs=100000000,
                                                    checkpoint_dir=FLAGS.eval_dir)

        # TODO: add batch_per_epoch and creating cycle for evaluate all images in db

        batch_per_epoch = int(dataset.num_samples['valid'] / FLAGS.batch_size)

        with session._tf_sess() as sess:
            i = 0
            full_eval_time = 0

            next_index = lambda step: (step * FLAGS.batch_size) % dataset.num_samples['train'] + 1

            matches = 0
            for step in range(batch_per_epoch):

                # # Get images for training
                # TODO: Check index
                time_start = time()
                index = next_index(step)
                images_batch, labels_batch = dataset.select_batch_by_index(table,
                                                                           index=index,
                                                                           batch_size=FLAGS.batch_size)
                db_time = time() - time_start

                # Train autoencoder
                list_acc_pos = sess.run(predictions, feed_dict={images_placeholder: images_batch})
                time_end = time()

                for batch in range(len(list_acc_pos)):
                    acc_pos = list_acc_pos[batch]
                    if labels_batch[batch][acc_pos] != 0.0:
                        matches += 1

                tf.logging.info(
                    'Data from db: {:.3f} sec/batch, Step: {}, ({:.3f} sec/step)'.format(
                            db_time,
                            step + 1,
                            time_end - time_start))

                full_eval_time += time_end - time_start
                i += 1

        tf.logging.info('Accuracy: {:.3f}'.format(matches / dataset.num_samples['valid']))

        dataset.close()

    full_eval_time = timedelta(seconds=full_eval_time)
    full_eval_time = datetime(1, 1, 1) + full_eval_time
    if full_eval_time.day - 1 != 0:
        tf.logging.info('Full eval time: %dd %dh %dm %ds', full_eval_time.day - 1, full_eval_time.hour,
                        full_eval_time.minute, full_eval_time.second)
    elif full_eval_time.hour != 0:
        tf.logging.info('Full eval time: %dh %dm %ds', full_eval_time.hour, full_eval_time.minute,
                        full_eval_time.second)
    elif full_eval_time.minute != 0:
        tf.logging.info('Full eval time: %dm %ds', full_eval_time.minute, full_eval_time.second)
    else:
        tf.logging.info('Full eval time: %ds', full_eval_time.second)


if __name__ == '__main__':
    tf.app.run()
