from __future__ import division, print_function, absolute_import

import time
from collections import defaultdict

import tensorflow as tf
from datasets import cifar10
from nets import mobilenet_v1
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim

model_load_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_mnist_rmsprop_1_epoch/model.ckpt-5000'
# model_save_path = '/tmp/tensorflow/train'
model_save_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_cifar10_ae/train_custom_script_not_ae'
dataset_dir = '/media/w_programs/NN_Database/data/cifar10/'
logs_every_n_step = 10

dataset = cifar10
batch_size = 10  # Number of samples in each batch
epoch_num = 1  # Number of epochs to train the network
lr = 0.001  # Learning rate
use_ae_vars = False
deconv = '_dconv2d'

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


def train_batch(sess, train_op, epoch, batch, logs_every_n_step=1):
    start_time = time.time()
    # total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
    _, c = sess.run([train_op, loss])
    time_elapsed = time.time() - start_time
    if batch_n % logs_every_n_step == 0:
        tf.logging.info(
            'Epoch={}, batch={}/{}, cost= {:.5f}, ({:.3f} sec/step)'.format((epoch + 1), batch + 1, batch_per_ep, c,
                                                                            time_elapsed))
    return c


# vars_load = {}
def init_fn():
    global_vars = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars_load = {}
    for var in global_vars:
        if variables_to_restore[var._shared_name] is not None:
            vars_load[var._shared_name] = variables_to_restore[var._shared_name]
    return slim.assign_from_values_fn(vars_load)


variables_to_restore = defaultdict(lambda: None)
if use_ae_vars:
    sess_load = tf.Session()
    with sess_load:
        saver_load = tf.train.import_meta_graph(model_load_path + '.meta')
        saver_load.restore(sess_load, tf.train.latest_checkpoint('/'.join(model_load_path.split('/')[:-1])))
        vars_load = sess_load.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in vars_load:
            variables_to_restore[var._shared_name] = var.eval()

with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    ######################
    # Select the dataset #
    ######################
    dataset = dataset.get_split(split_name='train',
                                dataset_dir=dataset_dir)
    images, labels = load_batch(dataset, batch_size, 32, 32, is_training=True)

    # calculate the number of batches per epoch
    batch_per_ep = dataset.num_samples // batch_size

    ae_outputs, end_points = mobilenet_v1.mobilenet_v1(images, num_classes=dataset.num_classes)

    # =================================================================================================
    # LOSS
    # =================================================================================================
    # loss = tf.reduce_mean(tf.square(end_points['AuxLogits'] - labels))
    loss = slim.losses.softmax_cross_entropy(ae_outputs, labels,
                                             label_smoothing=0.0, weights=0.4,
                                             scope='loss')
    # tf.reduce_mean(tf.square(ae_outputs - labels))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9)
    train_op = slim.learning.create_train_op(loss, optimizer)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for losses.
    summaries.add(tf.summary.scalar('loss', loss))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    if use_ae_vars:
        init = init_fn()
    else:
        init = None

    sv = tf.train.Supervisor(logdir=model_save_path, summary_op=summary_op, init_fn=init)

    with sv.managed_session() as sess_new:
        for ep_n in range(epoch_num):
            for batch_n in range(batch_per_ep):
                cost = train_batch(sess_new, train_op, ep_n, batch_n, logs_every_n_step=logs_every_n_step)

            save_path = sv.saver.save(sess_new, model_save_path + '/model.ckpt', global_step=batch_per_ep * (ep_n + 1))
            print('Saving model: {}'.format(save_path))
