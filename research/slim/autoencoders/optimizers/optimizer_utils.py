import numpy as np
from enum import Enum


class Formulas(Enum):
    golovko = 1
    hinton = 2


class LayerShapeType(Enum):
    NHWC = 1
    CHWN = 2


# Calc padding 'SAME'
# layer_type = x. Shape = NHWC
# layer_type = y. Shape = HWCN
def padding_fn(layer, w_shape, stride):
    layer_shape = layer.shape

    h = 1
    w = 2

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

    new_layer = np.zeros(shape=(layer_shape[0],
                                pad_top + layer_shape[1] + pad_bottom,
                                pad_left + layer_shape[2] + pad_right,
                                layer_shape[3]), dtype=np.float32)
    new_layer[:,
              pad_top: layer_shape[1] + pad_top,
              pad_left: layer_shape[2] + pad_left,
              :] = layer

    return new_layer
