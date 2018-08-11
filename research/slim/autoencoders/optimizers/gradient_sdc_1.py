import tensorflow as tf
import numpy as np
from math import e
from enum import Enum


# Производная активации
def d_activation(y, name):
    result = None
    if name == DActivation.relu:
        result = 1.0 if y > 0 else 0.0
    elif name == DActivation.leaky_relu:
        result = 1.0 if y > 0 else 0.2
    elif name == DActivation.sigmoid:
        result = y * (1.0 - y)
    elif name == DActivation.tanh:
        result = 1.0 - pow(y, 2)
    return result


class Formulas(Enum):
    golovko = 1
    hinton = 2


class DActivation(Enum):
    relu = 1
    leaky_relu = 2
    sigmoid = 3
    tanh = 4


class GradientSDC1:

    def __init__(self, grads, x, y, stride, padding, formulas=Formulas, d_activation_name=None,
                 ae_layer_prefix='recover'):
        self._grads = grads
        self._y = y
        self._x = x
        self._stride = stride
        self._padding = padding
        self._prefix = ae_layer_prefix
        self._act_name = d_activation_name
        self._d_act = d_activation if formulas == Formulas.golovko else lambda y, name: 1.0

    def grad_weight(self, grad, ye, xe):

        y = self._y
        x = self._x
        stride = self._stride
        act_name = self._act_name
        d_act = self._d_act

        x_shape = x[0].shape
        y_shape = y[0].shape
        w_shape = grad.shape

        grad_value = np.zeros(w_shape)

        for c in range(y_shape[2]):
            for k in range(y_shape[3]):
                for m in range(w_shape[0]):
                    for n in range(w_shape[1]):
                        w = []
                        for i in range(y_shape[0]):
                            for j in range(y_shape[1]):
                                for q in range(x_shape[0]):
                                    w.append(ye[i][j][c][k] * \
                                             x[1][q][i * stride + m][j * stride + n][c] * \
                                             d_act(y[1][i][j][c][k], act_name) + \
                                             xe[q][i * stride + m][j * stride + n][c] * \
                                             y[0][i][j][c][k] * \
                                             d_act(x[1][q][i * stride + m][j * stride + n][c], act_name))
                        grad_value[m][n][c][k] = np.sum(w)
        return grad_value

    def grad_biases_x(self, grads, grad_name, xe):
        y = self._y
        x = self._x
        stride = self._stride
        prefix = self._prefix
        act_name = self._act_name
        d_act = self._d_act

        x_shape = x[0].shape
        y_shape = y[0].shape

        # Get weights_shape for x1 layers
        w_shape = None
        for name in grads:
            if name.find(prefix) != -1 and name.find('weight') != -1:
                w_shape = grads[name].shape.as_list()
                break

        grad_value = np.zeros(grads[grad_name].shape)

        for q in range(x_shape[0]):
            t = []
            for c in range(x_shape[3]):
                for m in range(w_shape[0]):
                    for n in range(w_shape[1]):
                        for i in range(y_shape[0]):
                            for j in range(y_shape[1]):
                                t.append(xe[q][i * stride + m][j * stride + n][c] * \
                                         d_act(x[1][q][i * stride + m][j * stride + n][c], act_name))
            # grad_value.append(np.sum(t))
            grad_value[q] = np.sum(t)
        return grad_value

    def grad_biases_y(self, grad, ye):
        y = self._y
        act_name = self._act_name
        d_act = self._d_act

        y_shape = y[0].shape

        grad_value = np.zeros(grad.shape)

        for k in range(y_shape[3]):
            t = []
            for c in range(y_shape[2]):
                for i in range(y_shape[0]):
                    for j in range(y_shape[1]):
                        t.append(ye[i][j][c][k] * \
                                 d_act(y[1][i][j][c][k], act_name))
            grad_value[k] = np.sum(t)

        return grad_value

    def calc(self):
        grads = self._grads
        y = self._y
        x = self._x
        padding = self._padding
        prefix = self._prefix

        # Check layers shape
        assert x[0].shape == x[1].shape
        assert y[0].shape == y[1].shape
        assert len(y[0].shape) == len(x[0].shape) == 4

        ye = y[1] - y[0]
        xe = x[1] - x[0]

        grads_value = {}

        for grad_name in grads:

            if padding == 'SAME':
                """ Change x0, x1 layers """

            # Gradient for weights
            if grad_name.find('weight') != -1:
                grad_value = self.grad_weight(grads[grad_name], ye, xe)

            # Gradient for biases
            else:
                # Gradient for input_sdx_1 layer
                if grad_name.find(prefix) != -1:
                    grad_value = self.grad_biases_x(grads, grad_name, xe)
                # Gradient for output_sdc_0 layer
                else:
                    grad_value = self.grad_biases_y(grads[grad_name], ye)

            grads_value[grads[grad_name]] = grad_value

        return grads_value
