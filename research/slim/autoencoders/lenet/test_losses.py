import tensorflow as tf
from research.slim.autoencoders.lenet import lenet_bm
from tensorflow.contrib import slim

image = tf.placeholder(tf.float32, shape=(1, 28, 28, 3))
block_numbers = lenet_bm.lenet_bm.block_number
for block in range(block_numbers):
    output, end_points = lenet_bm.lenet_bm(image, train_block_num=block, sdc_num=1)

    op_list_no_optimizer = tf.get_default_graph().get_operations()

    loss_map = lenet_bm.lenet_model_losses(end_points, train_block_num=block, sdc_num=1)
    loss_list = []
    for loss in loss_map:
        loss_list.append(tf.reduce_sum(tf.divide(tf.square(loss_map[loss]['input'] - loss_map[loss]['output']),
                                                 tf.constant(2.0))))
    loss_op = tf.reduce_mean(loss_list)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, momentum=0.9, epsilon=1.0)
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    op_list_yes_optimizer = tf.get_default_graph().get_operations()
    print()
