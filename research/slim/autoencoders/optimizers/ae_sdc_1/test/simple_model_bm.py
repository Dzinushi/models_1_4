import tensorflow as tf
from tensorflow.contrib import slim
from collections import OrderedDict
from autoencoders.ae_utils import scope_with_sdc, set_model_losses, sdc_conv_block, sdc_fully_connected_block

model_scope = ''


def simple_model_bm(inputs, train_block_num=0, sdc_num=1):
    with tf.variable_scope(model_scope):
        layer_scope = 'input'
        net = inputs
        end_points = OrderedDict()
        end_points[scope_with_sdc(layer_scope, 0)] = net

        trainable_fn = lambda train_block_num, block_num: block_num == train_block_num

        pad = 'SAME'
        stride = 1

        # SDC Block 0
        if trainable_fn(train_block_num, 0):
            net, end_points = sdc_conv_block(end_points, net,
                                             num_outputs=2,
                                             num_sdc_outputs=1,
                                             kernel_size=[2, 2],
                                             padding=pad,
                                             sdc_num=sdc_num,
                                             scope_conv='conv1',
                                             scope_dconv='input')
            return net, end_points, pad, stride
        else:
            layer_scope = 'conv1'
            net = slim.conv2d(net, 2, [2, 2], padding=pad, trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)
            end_points[scope_with_sdc(layer_scope, 0)] = net

        end_points[scope_with_sdc('Flatten', 0)] = net = slim.flatten(net, scope='Flatten')

        pad = 'VALID'
        stride = 1

        # SDC Block 2
        net, end_points = sdc_fully_connected_block(end_points, net,
                                                    num_outputs=2,
                                                    num_sdc_outputs=int(
                                                        end_points[scope_with_sdc('Flatten', 0)].shape.as_list()[1]),
                                                    scope_fully_connected='fc2',
                                                    scope_dfully_connected='Flatten')
    return net, end_points, pad, stride


simple_model_bm.default_image_size = 2
simple_model_bm.block_number = 2


def simple_model_losses(end_points, train_block_num=0, sdc_num=1):
    layers_scope = {0: ['input', 'conv1'],
                    1: ['Flatten', 'fc2']}
    return set_model_losses(end_points, layers_scope[train_block_num], sdc_num)
