import tensorflow as tf
import numpy as np
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from research.slim.nets import lenet
from research.slim.autoencoders.utils import load_layer_vars
import time

slim = tf.contrib.slim

model_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_mnist_rmsprop_1_epoch/model.ckpt-1100'
model_save_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_mnist_1_epoch/'

batch_size = 50  # Number of samples in each batch
epoch_num = 1    # Number of epochs to train the network
lr = 0.0001       # Learning rate
dropout_keep_prob = 1.0
num_classes = 10
is_training = True
use_ae_vars = False

tf.set_random_seed(1)


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


def accuracy(predicted, labels, batch_size):
    sum = 0.0
    for i in range(batch_size):
        index = np.where(predicted[0][i] == predicted[0][i].max())[0][0]
        sum += abs(labels[i][index] - predicted[0][i].max())
    return sum / batch_size


def load_lenet_wb(load_graph, load_session, graph, session, scope='LeNet'):
    assign_layer = lambda layer_scope: load_layer_vars(
        l_graph=load_graph,
        l_session=load_session,
        graph=graph,
        session=session,
        scope=scope,
        layer_scope=layer_scope)

    assign_layer('conv1')
    assign_layer('conv2')
    assign_layer('fc3')


# Load model from checkpoint and removing autoencoder layers
old_sess = tf.Session()
with old_sess as sess:

    if use_ae_vars:
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/'.join(model_path.split('/')[:-1])))
        old_graph = tf.get_default_graph()

    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        # read MNIST dataset
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

        # calculate the number of batches per epoch
        batch_per_ep = mnist.train.num_examples // batch_size

        image = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
        labels = tf.placeholder(tf.float32, (None, 10))

        output, end_points = lenet.lenet(image, is_training=is_training, dropout_keep_prob=dropout_keep_prob)

        # calculate the loss and optimize the network
        loss = tf.reduce_mean(tf.square(output - labels))  # claculate the mean square error loss
        # loss = slim.losses.softmax_cross_entropy(
        #     ae_outputs, labels, weights=1.0)
        train_op = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)

        # initialize the network
        init = tf.global_variables_initializer()
        sess.run(init)

        # Load vars from pretrained ae_lenet
        if use_ae_vars:
            load_lenet_wb(old_graph, old_sess, new_graph, sess)
        old_sess.close()

        c = tf.placeholder(tf.float32)

        # Save graph
        writer = tf.summary.FileWriter(model_save_path, sess.graph)
        summary_loss = tf.summary.scalar('loss', loss)

        # batch_per_ep = 100
        for ep in range(epoch_num):  # epochs loop
            ep_time = 0.0
            for batch_n in range(batch_per_ep):  # batches loop
                batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
                batch_img = batch_img.reshape((-1, 28, 28, 1))  # reshape each sample to an (28, 28) image
                batch_img = resize_batch(batch_img)  # reshape the images to (32, 32)
                time_start = time.time()
                _, c, summary = sess.run([train_op, loss, summary_loss], feed_dict={image: batch_img, labels: batch_label})
                time_end = time.time() - time_start
                ep_time += time_end
                writer.add_summary(summary, global_step=(ep + 1) * (batch_n + 1))
                train_proc_str = '\rEpoch={}, batch={}/{}, cost= {:.5f}, time={:.3}s'.format(
                    (ep + 1), batch_n + 1, batch_per_ep, c, time_end)
                print(train_proc_str, end='')

            print('\nEpoch time {:.3}s'.format(ep_time))

            # test the trained network
            test_batch_per_ep = mnist.test.num_examples // batch_size
            total_acc = 0.0
            for batch_n in range(test_batch_per_ep):
                batch_img, batch_label = mnist.test.next_batch(batch_size)
                batch_img = resize_batch(batch_img)

                # Blocks
                eval_output = sess.run([end_points['Predictions']],
                                       feed_dict={image: batch_img})
                total_acc += accuracy(eval_output, batch_label, batch_size)

            print('Accuracy: %f' % (total_acc / test_batch_per_ep))

            # Save model
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_save_path + 'model.ckpt', global_step=batch_per_ep * (ep + 1))
            print('Saving model: {}'.format(save_path))

