from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from research.slim.autoencoders.lenet import lenet_bm

model_save_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_mnist_rmsprop_1_epoch/'

batch_size = 50  # Number of samples in each batch
epoch_num = 1  # Number of epochs to train the network
lr = 0.001  # Learning rate


def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
# ae_outputs = autoencoder_lenet_simple(ae_inputs)  # create the Autoencoder network
ae_outputs, end_points = lenet_bm.lenet_bm(ae_inputs, sdc_num=1)

# =================================================================================================
# LOSS
# =================================================================================================
#  loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
loss = tf.reduce_mean(tf.square(end_points['input'] - end_points['conv1_d'])) + \
       tf.reduce_mean(tf.square(end_points['pool1'] - end_points['conv2_d'])) + \
       tf.reduce_mean(tf.square(end_points['Flatten'] - end_points['fc3_d']))
train_op = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

# batch_per_ep = 100
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1))  # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)  # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            train_proc_str = '\rEpoch={}, batch={}/{}, cost= {:.5f}'.format((ep + 1), batch_n + 1, batch_per_ep, c)
            print(train_proc_str, end='')

        # Save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_save_path + 'model.ckpt', global_step=batch_per_ep)
        tf.summary.FileWriter(model_save_path, sess.graph)
        print('\nSaving model: {}'.format(save_path))

    # test the trained network
    batch_img, _ = mnist.test.next_batch(batch_size)
    batch_img = resize_batch(batch_img)

    # Blocks
    eval_input, eval_conv1_d, eval_pool1, eval_conv2_d = sess.run([end_points['input'], end_points['conv1_d'],
                                                                   end_points['pool1'], end_points['conv2_d']],
                                                                  feed_dict={ae_inputs: batch_img})

    # plot the reconstructed images and their ground truths (inputs)
    # Block 1
    if not os.path.exists(model_save_path + '/figures/'):
        os.mkdir(model_save_path + '/figures/')

    fig1 = plt.figure(1)
    plt.title('conv1_d')
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(eval_conv1_d[i, ..., 0], cmap='gray')
    fig1.savefig(model_save_path + '/figures/' + 'conv1_d')

    fig2 = plt.figure(2)
    plt.title('input')
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(eval_input[i, ..., 0], cmap='gray')
    fig2.savefig(model_save_path + '/figures/' + 'input')

    # Block 2
    fig3 = plt.figure(3)
    plt.title('conv2_d')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(eval_conv2_d[i, ..., 0], cmap='gray')
    fig3.savefig(model_save_path + '/figures/' + 'conv2_d')

    fig4 = plt.figure(4)
    plt.title('pool1')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(eval_pool1[i, ..., 0], cmap='gray')
    fig4.savefig(model_save_path + '/figures/' + 'pool1')
