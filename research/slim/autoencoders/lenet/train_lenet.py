from __future__ import division, print_function, absolute_import

import tensorflow as tf
from research.slim.autoencoders.optimizers import lenet_custom
from tensorflow.contrib import slim
from collections import defaultdict
from datasets import flowers
from time import time

from preprocessing import inception_preprocessing
from research.slim.autoencoders.optimizers.gradient_descent_optimizer_v2 import GradientDescentOptimizer_2

model_save_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_flowers_sgd_1_epoch/train_classificator_sgd_2'

batch_size = 1  # Number of samples in each batch
epoch_num = 1  # Number of epochs to train the network
lr = 0.001  # Learning rate
log_every_n_step = 100
max_step = 1000


tf.set_random_seed(777)


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

    image = inception_preprocessing.preprocess_image(image, height, width, is_training=is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)

    return images, labels


# Set autoencoder model
model_fn = lenet_custom.lenet

# Set image size
image_size = model_fn.default_image_size

# List pretrained vars for autoencoder train_block_number > 0
list_assign = None
# logdir_fn = lambda block_number: model_save_path + '/train_block_{}/'.format(block_number)
with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    database = flowers
    dataset = database.get_split('train',
                                 '/media/w_programs/NN_Database/data/flowers/')
    if max_step is None:
        max_step = dataset.num_samples // batch_size

    images, labels = load_batch(dataset, batch_size, image_size, image_size, is_training=True)
    outputs, end_points = model_fn(images, num_classes=database._NUM_CLASSES)

    # =================================================================================================
    # LOSS
    # =================================================================================================
    loss_op = tf.losses.softmax_cross_entropy(labels, outputs)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9)
    optimizer = GradientDescentOptimizer_2(learning_rate=lr, activation_name='leaky_relu')
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    # Create session using Supervisor
sv = tf.train.Supervisor(logdir=model_save_path,
                         save_model_secs=100000000,
                         graph=graph)

with sv.managed_session() as sess:
    # Initialize pretrained variables values
    total_loss = 0.0
    global_step = sess.run(sv.global_step)
    i = 0
    while global_step < max_step:
        time_start = time()
        loss = sess.run(train_op)
        time_end = time()
        total_loss += loss
        if (global_step + 1) % log_every_n_step == 0:
            tf.logging.info(
                'Step: {}, loss: {:.5f}, total_loss: {:.5f} ({:.3f} sec/step)'.format(global_step + 1,
                                                                                      loss,
                                                                                      total_loss / (i + 1),
                                                                                      time_end - time_start))
        global_step += 1
        i += 1
    checkpoint_path = model_save_path + '/model.ckpt'
    sv.saver.save(sess, save_path=checkpoint_path, global_step=max_step)
