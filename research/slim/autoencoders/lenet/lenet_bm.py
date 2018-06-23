import tensorflow as tf
from tensorflow.contrib import slim

model_scope = 'LeNet'


def lenet_bm(inputs, sdc_num=2):
    sdc_num += 1
    """SDC block 1 bolcman machine"""
    end_points = {'input': inputs}
    net = inputs
    net_d = None
    reuse = False
    with tf.variable_scope(model_scope):
        for i in range(sdc_num):
            # Encoder 1
            # 32 x 32 x 1
            end_points['conv1'] = net = slim.conv2d(net, 32, [5, 5], stride=1, reuse=reuse, scope='conv1')
            # Decoder 1
            # 32 x 32 x 32
            end_points['conv1_d'] = net_d = slim.conv2d_transpose(net, 3, [5, 5], stride=1, activation_fn=tf.nn.tanh,
                                                                  reuse=reuse,
                                                                  scope='conv1_d')
            if i + 1 < sdc_num:
                net = net_d
                reuse = True
            else:
                reuse = False

        # 32 x 32 x 32
        end_points['pool1'] = net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

        """SDC block 2 bolcman machine"""
        for i in range(sdc_num):
            # Encoder 2
            # 16 x 16 x 32
            end_points['conv2'] = net = slim.conv2d(net, 64, [5, 5], stride=1, reuse=reuse, scope='conv2')

            # Decoder 2
            # 16 x 16 x 64
            end_points['conv2_d'] = net_d = slim.conv2d_transpose(net, 32, [5, 5], stride=1, activation_fn=tf.nn.tanh,
                                                                  reuse=reuse,
                                                                  scope='conv2_d')
            if i + 1 < sdc_num:
                net = net_d
                reuse = True
            else:
                reuse = False

        # 16 x 16 x 64
        end_points['pool2'] = net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')

        # 8 x 8 x 64 (4096)
        end_points['Flatten'] = net = slim.flatten(net, scope='Flatten')

        """SDC block 3 bolcman machine"""
        for i in range(sdc_num):
            # Encoder 3
            # 1 x 1024
            end_points['fc3'] = net = slim.fully_connected(net, 1024, reuse=reuse, scope='fc3')

            # Decoder 4
            # 1 x 1024
            end_points['fc3_d'] = net_d = slim.fully_connected(net, 4096, activation_fn=tf.nn.tanh, reuse=reuse,
                                                               scope='fc3_d')

            if i + 1 < sdc_num:
                net = net_d
                reuse = True
            else:
                reuse = False

    return net, end_points
lenet_bm.default_image_size = 32


def lenet_bm_arg_scope(weight_decay=0.0):
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            activation_fn=tf.nn.relu) as sc:
        return sc


def get_main_ls():
    scope = model_scope + '/'
    list_ls = ['conv1', 'conv2', 'fc3']
    return [scope + ls for ls in list_ls]
