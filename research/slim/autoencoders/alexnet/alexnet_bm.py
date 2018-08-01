# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict
from autoencoders.ae_utils import sdc_conv_block, set_model_losses, scope_with_sdc
from tensorflow.contrib import slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

dconv = '_dconv2d'


def alexnet_v2(inputs,
               scope='alexnet_v2',
               sdc_num=1,
               train_block_num=0,
               channel=3):
    """AlexNet version 2.

    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224 or set
          global_pool=True. To use in fully convolutional mode, set
          spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        logits. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original AlexNet.)

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0
        or None).
      end_points: a dict of tensors with intermediate activations.
    """

    trainable_fn = lambda train_block_num, block_num: block_num == train_block_num

    with tf.variable_scope(scope, 'alexnet_v2', [inputs]):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        net = inputs
        end_points = OrderedDict()
        end_points[scope_with_sdc('input', 0)] = net

        # SDC 0
        if trainable_fn(train_block_num, 0):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=64,
                                             num_sdc_outputs=channel,
                                             kernel_size=[11, 11],
                                             stride=4,
                                             padding='VALID',
                                             sdc_num=sdc_num,
                                             scope_conv='conv1',
                                             scope_dconv='input')
            return net, end_points
        else:
            layer_scope = 'conv1'
            net = slim.conv2d(inputs, 64, [11, 11], 4, trainable=False, reuse=tf.AUTO_REUSE, padding='VALID',
                              scope=layer_scope)
            end_points[scope_with_sdc(layer_scope, 0)] = net

        layer_scope = 'pool1'
        end_points[scope_with_sdc(layer_scope, 0)] = net = slim.max_pool2d(net, [3, 3], 2, scope=layer_scope)

        # SDC 1
        if trainable_fn(train_block_num, 1):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=192,
                                             num_sdc_outputs=64,
                                             kernel_size=[5, 5],
                                             sdc_num=sdc_num,
                                             scope_conv='conv2',
                                             scope_dconv='pool1')
            return net, end_points
        else:
            layer_scope = 'conv2'
            net = slim.conv2d(net, 192, [5, 5], trainable=False, reuse=tf.AUTO_REUSE, scope='conv2')
            end_points[scope_with_sdc(layer_scope, 0)] = net

        layer_scope = 'pool2'
        end_points[scope_with_sdc(layer_scope, 0)] = net = slim.max_pool2d(net, [3, 3], 2, scope=layer_scope)

        # SDC 2
        if trainable_fn(train_block_num, 2):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=384,
                                             num_sdc_outputs=192,
                                             kernel_size=[3, 3],
                                             sdc_num=sdc_num,
                                             scope_conv='conv3',
                                             scope_dconv='pool2')
            return net, end_points
        else:
            layer_scope = 'conv3'
            end_points[scope_with_sdc(layer_scope, 0)] = net = \
                slim.conv2d(net, 384, [3, 3], trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)

        # SDC 3
        if trainable_fn(train_block_num, 3):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=384,
                                             num_sdc_outputs=384,
                                             kernel_size=[3, 3],
                                             sdc_num=sdc_num,
                                             scope_conv='conv4',
                                             scope_dconv='conv3')
            return net, end_points
        else:
            layer_scope = 'conv4'
            end_points[scope_with_sdc(layer_scope, 0)] = net = \
                slim.conv2d(net, 384, [3, 3], trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)

        # SDC 4
        if trainable_fn(train_block_num, 4):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=256,
                                             num_sdc_outputs=384,
                                             kernel_size=[3, 3],
                                             sdc_num=sdc_num,
                                             scope_conv='conv5',
                                             scope_dconv='conv4')
            return net, end_points
        else:
            layer_scope = 'conv5'
            end_points[scope_with_sdc(layer_scope, 0)] = net = \
                slim.conv2d(net, 256, [3, 3], trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)

        layer_scope = 'pool5'
        end_points[scope_with_sdc(layer_scope, 0)] = net = slim.max_pool2d(net, [3, 3], 2, scope=layer_scope)

        # SDC 5
        if trainable_fn(train_block_num, 5):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=4096,
                                             num_sdc_outputs=256,
                                             kernel_size=[5, 5],
                                             sdc_num=sdc_num,
                                             padding='VALID',
                                             weights_initializer=trunc_normal(0.005),
                                             biases_initializer=tf.constant_initializer(0.1),
                                             scope_conv='fc6',
                                             scope_dconv='pool5')
            return net, end_points
        else:
            layer_scope = 'fc6'
            end_points[scope_with_sdc(layer_scope, 0)] = net = \
                slim.conv2d(net, 4096, [5, 5], trainable=False, reuse=tf.AUTO_REUSE, padding='VALID', scope=layer_scope)

        # SDC 6
        if trainable_fn(train_block_num, 6):
            net, end_points = sdc_conv_block(end_points,
                                             net,
                                             num_outputs=4096,
                                             num_sdc_outputs=4096,
                                             kernel_size=[1, 1],
                                             sdc_num=sdc_num,
                                             weights_initializer=trunc_normal(0.005),
                                             biases_initializer=tf.constant_initializer(0.1),
                                             scope_conv='fc7',
                                             scope_dconv='fc6')
            return net, end_points
        else:
            layer_scope = 'fc7'
            end_points[scope_with_sdc(layer_scope, 0)] = net = \
                slim.conv2d(net, 4096, [1, 1], trainable=False, reuse=tf.AUTO_REUSE, scope=layer_scope)

        return net, end_points


alexnet_v2.default_image_size = 223
alexnet_v2.block_number = 7


def alexnet_model_losses(end_points, train_block_num=0, sdc_num=1):
    layers_scope = {0: ['input', 'conv1'],
                    1: ['pool1', 'conv2'],
                    2: ['pool2', 'conv3'],
                    3: ['conv3', 'conv4'],
                    4: ['conv4', 'conv5'],
                    5: ['pool5', 'fc6'],
                    6: ['fc6', 'fc7']}

    return set_model_losses(end_points, layers_scope[train_block_num], sdc_num)
