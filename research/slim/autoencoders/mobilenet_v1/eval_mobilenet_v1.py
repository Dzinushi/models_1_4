from __future__ import division, print_function, absolute_import

import time

import tensorflow as tf
from datasets import cifar10
from nets import mobilenet_v1
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim

model_load_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_cifar10_ae/train_custom_script_not_ae/model.ckpt-5000'
# model_load_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_cifar10_ae/train_custom_script/'
dataset_dir = '/media/w_programs/NN_Database/data/cifar10/'
logs_every_n_step = 100

dataset = cifar10
batch_size = 1  # Number of samples in each batch
epoch_num = 1  # Number of epochs to train the network
lr = 0.001  # Learning rate
use_ae_vars = True
deconv = '_dconv2d'
test_percent = 0.1

tf.set_random_seed(777)


def load_batch(dataset, batch_size, height=224, width=224, is_training=False):
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


def train_batch(sess, train_op, epoch, batch, logs_every_n_step=1):
    start_time = time.time()
    # total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
    _, c = sess.run([train_op, loss])
    time_elapsed = time.time() - start_time
    tf.logging.info(
        'Epoch={}, batch={}/{}, cost= {:.5f}, ({:.3f} sec/step)'.format((epoch + 1), batch + 1, batch_per_ep, c,
                                                                        time_elapsed))
    return c


def accuracy(predicted, labels, batch_size):
    sum = 0.0
    for i in range(batch_size):
        max = predicted[0].max()
        index_max = -1
        for index in range(len(predicted[0])):
            if predicted[0][index] == max:
                index_max = index
                break
        sum += abs(labels[i][index_max] - max)
    return sum / batch_size


def test_batch(sess, end_points, batch_label, batch_n, logs_every_n_step=1):
    start_time = time.time()
    # total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
    eval_output = sess.run(end_points['Predictions'])
    time_elapsed = time.time() - start_time
    acc = accuracy(eval_output, sess.run(batch_label), batch_size)
    _, c = sess.run([train_op, loss])
    return acc, time_elapsed


with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    ######################
    # Select the dataset #
    ######################
    dataset = dataset.get_split(split_name='test',
                                dataset_dir=dataset_dir)
    images, labels = load_batch(dataset, batch_size, 32, 32, is_training=False)

    # calculate the number of batches per epoch
    batch_per_ep = int(dataset.num_samples // batch_size * test_percent)

    ae_outputs, end_points = mobilenet_v1.mobilenet_v1(images, num_classes=dataset.num_classes, is_training=False)

    # =================================================================================================
    # LOSS
    # =================================================================================================
    loss = tf.reduce_mean(tf.square(ae_outputs - labels))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9)
    train_op = slim.learning.create_train_op(loss, optimizer)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for losses.
    summaries.add(tf.summary.scalar('loss', loss))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # if use_ae_vars:
    #     init = init_fn()
    # else:
    #     init = None

    # slim.learning.train(train_op,
    #                     logdir=model_save_path,
    #                     number_of_steps=batch_per_ep,
    #                     init_fn=init,
    #                     summary_op=summary_op)

    sv = tf.train.Supervisor(logdir=model_load_path, summary_op=summary_op, saver=None)

    with sv.managed_session() as sess_new:
        total_acc = 0.0
        for ep_n in range(epoch_num):
            for batch_n in range(batch_per_ep):
                acc, time_elapsed = test_batch(sess_new, end_points, labels, batch_n, logs_every_n_step=logs_every_n_step)
                total_acc += acc
                if batch_n % logs_every_n_step == 0:
                    print('{}) Total accuracy: {:.3f} ({:.3f} sec/step)'.format(batch_n + 1, total_acc / (batch_n + 1), time_elapsed))
