from collections import defaultdict
from tensorflow.contrib import slim
from tensorflow.python.ops import nn
import tensorflow as tf

DCONV = '_dconv2d'


def is_reuse(net, net_d, index=0, sdc_num=2):
    if index + 1 < sdc_num:
        net = net_d
    return net


def scope_with_sdc(layer_scope, index):
    return layer_scope + '_sdc_' + str(index)


def sdc_conv_block(end_points,
                   input,
                   num_outputs,
                   num_sdc_outputs,
                   kernel_size,
                   activation=nn.relu,
                   activation_sdc=nn.relu,
                   stride=1,
                   padding='SAME',
                   sdc_num=1,
                   trainable=True,
                   scope_conv='',
                   scope_dconv=''):
    assert sdc_num > 0
    sdc_num += 1
    net = input
    net_d = None
    for i in range(sdc_num):

        # Compress layer
        net = slim.conv2d(inputs=net,
                          num_outputs=num_outputs,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          activation_fn=activation,
                          reuse=tf.AUTO_REUSE,
                          trainable=trainable,
                          scope=scope_conv)
        end_points[scope_with_sdc(scope_conv, i)] = net

        # Recover layer
        if i + 1 < sdc_num:
            net_d = slim.conv2d_transpose(inputs=net,
                                          num_outputs=num_sdc_outputs,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          activation_fn=activation_sdc,
                                          reuse=tf.AUTO_REUSE,
                                          trainable=trainable,
                                          scope=scope_dconv + DCONV)
            end_points[scope_with_sdc(scope_dconv, i + 1)] = net_d
        net = is_reuse(net, net_d, i, sdc_num)

    return net, end_points


def sdc_fully_connected_block(end_points,
                              input,
                              num_outputs,
                              num_sdc_outputs,
                              activation=nn.relu,
                              activation_sdc=nn.relu,
                              sdc_num=1,
                              trainable=True,
                              scope_fully_connected='',
                              scope_dfully_connected=''):
    assert sdc_num > 0
    sdc_num += 1
    net = input
    net_d = None
    for i in range(sdc_num):

        # Compress layer
        net = slim.fully_connected(inputs=net,
                                   num_outputs=num_outputs,
                                   activation_fn=activation,
                                   reuse=tf.AUTO_REUSE,
                                   trainable=trainable,
                                   scope=scope_fully_connected)
        end_points[scope_with_sdc(scope_fully_connected, i)] = net

        # Recover layer
        if i + 1 < sdc_num:
            net_d = slim.fully_connected(inputs=net,
                                         num_outputs=num_sdc_outputs,
                                         activation_fn=activation_sdc,
                                         reuse=tf.AUTO_REUSE,
                                         trainable=trainable,
                                         scope=scope_dfully_connected + DCONV)
            end_points[scope_with_sdc(scope_dfully_connected, i + 1)] = net_d
        net = is_reuse(net, net_d, i, sdc_num)

    return net, end_points


"""Append '_sdc_N' or '_dconv2d_sdc_N' to layer_scope value to check (where N=(0, 1, 2 .. N)"""
def set_model_losses(end_points, layers_scope, sdc_num=1):
    index = 0
    add_loss = lambda input_value, output_value: {'input': input_value, 'output': output_value}
    model_losses = defaultdict(lambda: None)

    for layer_name in layers_scope:
        num_layer_losses = 0
        for end_point in end_points:
            if end_point.startswith(layer_name):
                num_layer_losses += 1
        if num_layer_losses % 2 != 0:
            ValueError('Error setting loss value: odd number of losses: %d' % num_layer_losses)
            return 1
        # scope = '/'.join(next(iter(layer_losses.keys())).split('/')[:-1]) + '/'
        scope = ''
        for i in range(sdc_num):
            input_scope = scope + scope_with_sdc(layer_name, i)
            output_scope = scope + scope_with_sdc(layer_name + DCONV, i + 1)
            if end_points[output_scope] is None:
                del end_points[output_scope]
                output_scope = scope + scope_with_sdc(layer_name, i + 1)
            if end_points[input_scope] is not None and end_points[output_scope] is not None:
                model_losses[index] = add_loss(end_points[input_scope], end_points[output_scope])
            else:
                ValueError(input_scope + ' or ' + output_scope + ' is not found in "end_points"')
                return 1
            index += 1

    return model_losses
