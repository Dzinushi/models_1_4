import tensorflow as tf
from tensorflow.contrib import slim
from collections import defaultdict
from autoencoders.ae_utils import scope_with_sdc, DCONV, set_model_losses, sdc_conv_block, sdc_fully_connected_block

model_scope = 'LeNet'
dconv = DCONV


def lenet_bm(inputs, sdc_num=1):
    """SDC block 1 bolcman machine"""
    net = inputs
    with tf.variable_scope(model_scope):
        layer_scope = 'input'
        end_points = defaultdict(lambda: None)
        end_points[scope_with_sdc(layer_scope, 0)] = inputs

        # SDC Block 1
        net, end_points = sdc_conv_block(end_points, net,
                                         num_outputs=32,
                                         num_sdc_outputs=3,
                                         kernel_size=[5, 5],
                                         sdc_num=sdc_num,
                                         scope_conv='conv1',
                                         scope_dconv='input')

        end_points[scope_with_sdc('pool1', 0)] = net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

        # SDC Block 2
        net, end_points = sdc_conv_block(end_points, net,
                                         num_outputs=64,
                                         num_sdc_outputs=32,
                                         kernel_size=[5, 5],
                                         sdc_num=sdc_num,
                                         scope_conv='conv2',
                                         scope_dconv='pool1')

        end_points[scope_with_sdc('pool2', 0)] = net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
        end_points[scope_with_sdc('Flatten', 0)] = net = slim.flatten(net, scope='Flatten')

        # SDC Block 3
        net, end_points = sdc_fully_connected_block(end_points, net,
                                                    num_outputs=1024,
                                                    num_sdc_outputs=int(end_points[scope_with_sdc('Flatten', 0)].shape.as_list()[1]),
                                                    scope_fully_connected='fc3',
                                                    scope_dfully_connected='Flatten')
    return net, end_points
lenet_bm.default_image_size = 28


def lenet_model_losses(end_points, sdc_num=1):
    layers_scope = ['input', 'conv1', 'pool1', 'conv2', 'Flatten', 'fc3']
    return set_model_losses(end_points, layers_scope, sdc_num)
