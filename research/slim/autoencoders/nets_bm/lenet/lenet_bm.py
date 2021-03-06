import tensorflow as tf
from tensorflow.contrib import slim
from collections import OrderedDict
from autoencoders.ae_utils import scope_with_sdc, set_model_losses, sdc_conv_block, sdc_fully_connected_block

model_scope = 'LeNet'


# Example activation_fn: tf.nn.relu
def lenet_bm(inputs, train_block_num=0, sdc_num=1, stride=1, activation_fn=tf.nn.relu):
    with tf.variable_scope(model_scope):
        layer_scope = 'input'
        net = inputs
        end_points = OrderedDict()
        end_points[scope_with_sdc(layer_scope, 0)] = net

        trainable_fn = lambda train_block_num, block_num: block_num == train_block_num

        pad = 'SAME'

        # SDC Block 0
        if trainable_fn(train_block_num, 0):
            net, end_points = sdc_conv_block(end_points, net,
                                             num_outputs=32,
                                             num_sdc_outputs=3,
                                             kernel_size=[5, 5],
                                             sdc_num=sdc_num,
                                             activation=activation_fn,
                                             activation_sdc=activation_fn,
                                             scope_conv='conv1',
                                             scope_dconv='input')
            return net, end_points, pad, stride
        else:
            layer_scope = 'conv1'
            net = slim.conv2d(net, 32, [5, 5], activation_fn=activation_fn, trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)
            end_points[scope_with_sdc(layer_scope, 0)] = net

        end_points[scope_with_sdc('pool1', 0)] = net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

        # SDC Block 1
        if trainable_fn(train_block_num, 1):
            net, end_points = sdc_conv_block(end_points, net,
                                             num_outputs=64,
                                             num_sdc_outputs=32,
                                             kernel_size=[5, 5],
                                             sdc_num=sdc_num,
                                             activation=activation_fn,
                                             activation_sdc=activation_fn,
                                             scope_conv='conv2',
                                             scope_dconv='pool1')
            return net, end_points, pad, stride
        else:
            layer_scope = 'conv2'
            net = slim.conv2d(net, 64, [5, 5], activation_fn=activation_fn, trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)
            end_points[scope_with_sdc(layer_scope, 0)] = net

        end_points[scope_with_sdc('pool2', 0)] = net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
        end_points[scope_with_sdc('Flatten', 0)] = net = slim.flatten(net, scope='Flatten')

        pad = 'VALID'

        # SDC Block 2
        net, end_points = sdc_fully_connected_block(end_points, net,
                                                    num_outputs=1024,
                                                    num_sdc_outputs=int(
                                                        end_points[scope_with_sdc('Flatten', 0)].shape.as_list()[1]),
                                                    activation=activation_fn,
                                                    activation_sdc=activation_fn,
                                                    scope_fully_connected='fc3',
                                                    scope_dfully_connected='Flatten')
    return net, end_points, pad, stride


lenet_bm.default_image_size = 28
lenet_bm.block_number = 3


def lenet_model_losses(end_points, train_block_num=0, sdc_num=1):
    layers_scope = {0: ['input', 'conv1'],
                    1: ['pool1', 'conv2'],
                    2: ['Flatten', 'fc3']}
    return set_model_losses(end_points, layers_scope[train_block_num], sdc_num)
