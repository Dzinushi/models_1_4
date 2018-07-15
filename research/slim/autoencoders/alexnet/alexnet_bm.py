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
from collections import defaultdict
from autoencoders.ae_utils import is_reuse

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

dconv = '_dconv2d'


def alexnet_v2(inputs,
               scope='alexnet_v2',
               sdc_num=1,
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
    sdc_num += 1

    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.conv2d_transpose],
                            outputs_collections=[end_points_collection]):
            net = None
            reuse = False
            for i in range(sdc_num):
                net = slim.conv2d(inputs, 64, [11, 11], 4, reuse=reuse, padding='VALID', scope='conv1')
                net_d = slim.conv2d_transpose(net, channel, [11, 11], 4, reuse=reuse,
                                              padding='VALID', scope='conv1' + dconv)
                net, reuse = is_reuse(net, net_d, i, sdc_num)

            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

            for i in range(sdc_num):
                net = slim.conv2d(net, 192, [5, 5], reuse=reuse, scope='conv2')
                net_d = slim.conv2d_transpose(net, 64, [5, 5], reuse=reuse, scope='conv2' + dconv)
                net, reuse = is_reuse(net, net_d, i, sdc_num)

            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

            for i in range(sdc_num):
                net = slim.conv2d(net, 384, [3, 3], reuse=reuse, scope='conv3')
                net_d = slim.conv2d_transpose(net, 192, [3, 3], reuse=reuse, scope='conv3' + dconv)
                net, reuse = is_reuse(net, net_d, i, sdc_num)

            for i in range(sdc_num):
                net = slim.conv2d(net, 384, [3, 3], reuse=reuse, scope='conv4')
                net_d = slim.conv2d_transpose(net, 384, [3, 3], reuse=reuse, scope='conv4' + dconv)
                net, reuse = is_reuse(net, net_d, i, sdc_num)

            for i in range(sdc_num):
                net = slim.conv2d(net, 256, [3, 3], reuse=reuse, scope='conv5')
                net_d = slim.conv2d_transpose(net, 384, [3, 3], reuse=reuse, scope='conv5' + dconv)
                net, reuse = is_reuse(net, net_d, i, sdc_num)

            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                for i in range(sdc_num):
                    net = slim.conv2d(net, 4096, [5, 5], reuse=reuse, padding='VALID', scope='fc6')
                    net_d = slim.conv2d_transpose(net, 256, [5, 5], reuse=reuse, padding='VALID', scope='fc6' + dconv)
                    net, reuse = is_reuse(net, net_d, i, sdc_num)

                for i in range(sdc_num):
                    net = slim.conv2d(net, 4096, [1, 1], reuse=reuse, scope='fc7')
                    net_d = slim.conv2d_transpose(net, 4096, [1, 1], reuse=reuse, scope='fc7' + dconv)
                    net, reuse = is_reuse(net, net_d, i, sdc_num)

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                end_points['alexnet_v2/input'] = inputs
            return net, end_points


alexnet_v2.default_image_size = 224


def get_loss_layer_names():
    index = -1
    next_index = lambda x: x + 1
    add = lambda input_value, output_value: {'input': 'alexnet_v2/' + input_value,
                                             'output': 'alexnet_v2/' + output_value}

    layer_names = defaultdict(lambda var: None)
    layer_names[next_index(index)] = add('input', 'conv1' + dconv)
    layer_names[next_index(index)] = add('pool1', 'conv2' + dconv)
    layer_names[next_index(index)] = add('pool2', 'conv3' + dconv)
    layer_names[next_index(index)] = add('conv3', 'conv4' + dconv)
    layer_names[next_index(index)] = add('conv4', 'conv5' + dconv)
    layer_names[next_index(index)] = add('pool5', 'fc6' + dconv)
    layer_names[next_index(index)] = add('fc6', 'fc7' + dconv)

    return layer_names
