from __future__ import division, print_function, absolute_import

import tensorflow as tf
from research.slim.autoencoders.lenet import lenet_bm
from tensorflow.contrib import slim
from collections import defaultdict
from datasets import flowers
from time import time

from preprocessing import inception_preprocessing
from research.slim.autoencoders.optimizers.sgd_sdc_1 import GradientDescentOptimizerSDC1

model_save_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_flowers_sgd_sdc_1_epoch_1/train_ae'

batch_size = 1  # Number of samples in each batch
epoch_num = 1  # Number of epochs to train the network
lr = 0.001  # Learning rate
log_every_n_step = 100
max_step = 500


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


def init_fn(list_pretrained_vars):
    global_vars = tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    list_assign_op = []
    for var in global_vars:
        if list_pretrained_vars[var._shared_name] is not None:
            list_assign_op.append(var.assign(list_pretrained_vars[var._shared_name]))
    return list_assign_op


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
ae_fn = lenet_bm.lenet_bm

# Set image size
image_size = ae_fn.default_image_size

# Set autoencoder trainable block counts
block_count = ae_fn.block_number

# List pretrained vars for autoencoder train_block_number > 0
list_assign = None
for train_block_number in range(block_count):
    logdir_fn = lambda block_number: model_save_path + '/train_block_{}/'.format(block_number)
    with tf.Graph().as_default() as graph:

        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = flowers.get_split('train',
                                    '/media/w_programs/NN_Database/data/flowers/')
        if max_step is None:
            max_step = dataset.num_samples // batch_size

        images, labels = load_batch(dataset, batch_size, image_size, image_size, is_training=True)
        ae_outputs, end_points = ae_fn(images, train_block_num=train_block_number, sdc_num=1)

        if train_block_number > 0:
            checkpoint_path = tf.train.latest_checkpoint(logdir_fn(train_block_number - 1))
            list_pretrained_vars = load_vars_from_path(checkpoint_path)
            list_assign = init_fn(list_pretrained_vars)

        # =================================================================================================
        # LOSS
        # =================================================================================================
        loss_map = lenet_bm.lenet_model_losses(end_points, train_block_number, sdc_num=1)
        list_losses = []
        for loss in loss_map[train_block_number]:
            list_losses.append(tf.reduce_sum(tf.divide(tf.square(loss_map[loss]['input'] - loss_map[loss]['output']),
                                                       tf.constant(2.0))))
        loss_op = tf.reduce_mean(list_losses)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9)
        optimizer = GradientDescentOptimizerSDC1(learning_rate=lr, activation_name='relu', loss_map=loss_map)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(loss_op, optimizer)

    # Create session using Supervisor
    sv = tf.train.Supervisor(logdir=logdir_fn(train_block_number),
                             save_model_secs=100000000,
                             graph=graph)

    with sv.managed_session() as sess:
        # Initialize pretrained variables values
        if list_assign is not None:
            sess.run(list_assign)
            tf.logging.info('Autoencoder pretrained variables successfully recovered!')
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
        checkpoint_path = logdir_fn(train_block_number) + 'model.ckpt'
        sv.saver.save(sess, save_path=checkpoint_path, global_step=max_step)
#         def save_picture(data, title):
#             fig1 = plt.figure(1)
#             plt.title(title)
#             if len(data.shape) == 4:
#                 data = data[:, :, :, :1]
#                 data = np.squeeze(data)
#             is_shape_ok = valid_imshow_data(data)
#             if not is_shape_ok:
#                 return -1
#             if data.max() == 0.0:
#                 print('{} max value is 0.0'.format(title))
#             else:
#                 multiplier = 1.0 / data.max()
#                 data = data * multiplier
#                 print('{} = {:.3f}'.format(title, multiplier))
#             plt.imshow(data, cmap='gray')
#             fig1.savefig(model_save_path + '/figures/' + title)
#
#
#         save_picture(input_0, 'input_sdc_0')  # 28 x 28
#         save_picture(input_1, 'input_sdc_1')
#         save_picture(conv_1_0, 'conv1_sdc_0')
#         save_picture(conv_1_1, 'conv1_sdc_1')
#         # save_picture(e_pool1_sdc_0, 'pool1_sdc_0')
#         # save_picture(e_pool1_sdc_1, 'pool1_sdc_1')
#         # save_picture(e_conv2_sdc_0, 'conv2_sdc_0')
#         # save_picture(e_conv2_sdc_1, 'conv2_sdc_1')
#         # flatten_0 = np.reshape(flatten_0, [56, 56])
#         # save_picture(flatten_0, 'flatten_sdc_0')
#         # flatten_1 = np.reshape(flatten_1, [56, 56])
#         # save_picture(flatten_1, 'flatten_sdc_1')
#         #
#         # fc3_0 = np.reshape(fc3_0, [32, 32])
#         # save_picture(fc3_0, 'fc3_sdc_0')
#         # fc3_1 = np.reshape(fc3_1, [32, 32])
#         # save_picture(fc3_1, 'fc3_sdc_1')
#
# # print(e_conv1_sdc_0)
