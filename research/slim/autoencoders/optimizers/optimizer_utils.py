import numpy as np
from enum import Enum


class Formulas(Enum):
    golovko = 1
    hinton = 2


class LayerShapeType(Enum):
    NHWC = 1
    CHWN = 2


# Производная активации
def d_activation_fn(y, name):
    result = None
    if name == 'relu':
        result = 1.0 if y > 0 else 0.0
    elif name == 'leakyrelu':
        result = 1.0 if y > 0 else 0.2
    elif name == 'sigmoid':
        result = y * (1.0 - y)
    elif name == 'tanh':
        result = 1.0 - pow(y, 2)
    return result


# @vectorize(["float32(float32)"], target='cuda')
def d_relu_cuda(y):
    y[y > 0] = 1.0
    y[y < 0] = 0.0
    return y


# @vectorize(["float32(float32)"], target='cuda')
def d_leakyrelu_cuda(y):
    y[y > 0] = 1.0
    y[y < 0] = 0.0
    return y


# @vectorize(["float32(float32)"], target='cuda')
def d_sigmoid_cuda(y):
    return y * (1.0 - y)


# @vectorize(["float32(float32)"], target='cuda')
def d_tanh_cuda(y):
    return 1.0 - pow(y, 2)


d_act_dic = {'relu': d_relu_cuda,
             'leakyrelu': d_leakyrelu_cuda,
             'sigmoid': d_sigmoid_cuda,
             'tanh': d_tanh_cuda}


# First input layer must contained 'input' in names. It has another layer shape type
def layer_shape_type(layer_tensor):
    if layer_tensor.name.find('input'):
        return LayerShapeType.NHWC
    else:
        return LayerShapeType.CHWN


# Calc padding 'SAME'
# layer_type = x. Shape = NHWC
# layer_type = y. Shape = HWCN
def padding_fn(layer, w_shape, stride, layer_type):
    layer_shape = layer.shape

    if layer_type == 'x':
        h = 1
        w = 2
    elif layer_type == 'y':
        h = 0
        w = 1
    else:
        raise ValueError('Undefind layer type: {}. Must be "x" or "y"'.format(layer_type))

    if layer_shape[h] % stride == 0:
        pad_along_height = w_shape[0] - stride
    else:
        pad_along_height = w_shape[0] - layer_shape[h] % stride

    if layer_shape[w] % stride == 0:
        pad_along_width = w_shape[1] - stride
    else:
        pad_along_width = w_shape[1] - layer_shape[w] % stride

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    if layer_type == 'x':
        new_layer = np.zeros(shape=(layer_shape[0],
                                    pad_top + layer_shape[1] + pad_bottom,
                                    pad_left + layer_shape[2] + pad_right,
                                    layer_shape[3]), dtype=np.float32)
        new_layer[:,
                  pad_top: layer_shape[1] + pad_top,
                  pad_left: layer_shape[2] + pad_left,
                  :] = layer
    else:
        new_layer = np.zeros(shape=(pad_top + layer_shape[0] + pad_bottom,
                                    pad_left + layer_shape[1] + pad_right,
                                    layer_shape[2],
                                    layer_shape[3]), dtype=np.float32)
        new_layer[pad_top: layer_shape[0] + pad_top,
                  pad_left: layer_shape[1] + pad_left,
                  :,
                  :] = layer

    return new_layer
