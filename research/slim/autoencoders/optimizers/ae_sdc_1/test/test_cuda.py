import numpy as np
from timeit import default_timer as timer
# from numba import vectorize, jit, cuda
import numpy as np
from math import sqrt
from research.slim.autoencoders.optimizers.optimizer_utils import d_act_dic
from research.slim.autoencoders.optimizers.optimizer_utils import Formulas


# CPU version
def vector_add(a, b):
    return a + b


# GPU version
# @vectorize(['float32(float32, float32)'], target='cuda')
def vector_add_gpu(a, b):
    return a + b


def multiply(a, b):
    return a * b


# @vectorize(['float32(float32, float32)'], target='cpu')
def multiply_numba(a, b):
    return a * b


def weights_update(x0, x1, y0, y1, w_shape, stride):
    y_shape = y0.shape
    x_shape = x0.shape
    ye = y1 - y0
    xe = x1 - x0

    d_act = lambda var: 1.0

    grad = np.zeros(shape=w_shape)

    if len(y_shape) == 4:
        for c in range(w_shape[2]):
            for k in range(w_shape[3]):
                for m in range(w_shape[0]):
                    for n in range(w_shape[1]):
                        w = []
                        for i in range(y_shape[1]):
                            for j in range(y_shape[2]):
                                for q in range(x_shape[0]):
                                    w.append(ye[q][i][j][k] * \
                                             x1[q][i * stride + m][j * stride + n][c] * d_act(y1[q][i][j][k]) + \
                                             xe[q][i * stride + m][j * stride + n][c] * \
                                             y0[q][i][j][k] * d_act(x1[q][i * stride + m][j * stride + n][c]))
                        grad[m][n][c][k] = np.sum(w)
        grad /= y_shape[1] * y_shape[2]
    elif len(y_shape) == 2:

        x0_temp = np.sum(x0, axis=0)
        x1_temp = np.sum(x1, axis=0)

        y0_temp = np.sum(y0, axis=0)
        y1_temp = np.sum(y1, axis=0)

        if w_shape[0] == x0.shape[1]:
            for i in range(x_shape[1]):
                for j in range(y_shape[1]):
                    grad[i][j] = (y1_temp[j] - y0_temp[j]) * x1_temp[i] * d_act(y1_temp[j]) + \
                                 (x1_temp[i] - x0_temp[i]) * y0_temp[j] * d_act(x1_temp[i])
        else:
            for i in range(x_shape[1]):
                for j in range(y_shape[1]):
                    grad[j][i] = (y1_temp[j] - y0_temp[j]) * x1_temp[i] * d_act(y1_temp[j]) + \
                                 (x1_temp[i] - x0_temp[i]) * y0_temp[j] * d_act(x1_temp[i])
    return grad


def weight_update_conv_np_stride_1(x0, x1, y0, y1, w_shape):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

    d_act = lambda var: 1.0

    for m in range(w_shape[0]):
        for n in range(w_shape[1]):
            for q in range(w_shape[2]):
                # Get part of x by w(m,n)
                x0_temp = x0[:, m:y_shape[1] + m, n:y_shape[2] + n, q]
                x1_temp = x1[:, m:y_shape[1] + m, n:y_shape[2] + n, q]

                # Summary x by batch size
                x0_temp = np.sum(x0_temp, axis=0)
                x1_temp = np.sum(x1_temp, axis=0)

                y0_temp = np.sum(y0, axis=0)
                y1_temp = np.sum(y1, axis=0)

                # Resize x
                x0_temp = np.resize(x0_temp, new_shape=(x0_temp.shape[0], x0_temp.shape[1], 1))
                x1_temp = np.resize(x1_temp, new_shape=x0_temp.shape)

                result = (x1_temp - x0_temp) * y0_temp * d_act(x1_temp) + \
                         (y1_temp - y0_temp) * x1_temp * d_act(y1_temp)

                grad[m, n, q, :] = np.sum(result, axis=(0, 1))

    # WARNING! Need to append division by batch_size
    grad /= y_shape[1] * y_shape[2]
    return grad


def weight_update_conv_np_stride_n(x0, x1, y0, y1, w_shape, stride):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

    d_act = lambda var: 1.0

    for m in range(w_shape[0]):
        for n in range(w_shape[1]):
            for q in range(w_shape[2]):
                # Get part of x by w(m,n)
                x0_temp = x0[:, m::stride, n::stride, q]
                x0_temp = x0_temp[:, :y_shape[1], :y_shape[2]]

                x1_temp = x1[:, m::stride, n::stride, q]
                x1_temp = x1_temp[:, :y_shape[1], :y_shape[2]]

                # Summary x and y by q (batch number)
                x0_temp = np.sum(x0_temp, axis=0)
                x1_temp = np.sum(x1_temp, axis=0)

                y0_temp = np.sum(y0, axis=0)
                y1_temp = np.sum(y1, axis=0)

                # Resize x
                x0_temp = np.resize(x0_temp, new_shape=(1, x0_temp.shape[0], x0_temp.shape[1], 1))
                x1_temp = np.resize(x1_temp, new_shape=(x0_temp.shape))

                result = (x1_temp - x0_temp) * y0_temp * d_act(x1_temp) + \
                         (y1_temp - y1_temp) * x1_temp * d_act(y1_temp)

                grad[m, n, q, :] = np.sum(result, axis=(0, 1))

    # WARNING! Need to append division by batch_size
    grad /= y_shape[1] * y_shape[2]
    return grad


def weight_update_fc_np(x0, x1, y0, y1, w_shape):
    grad = np.zeros(shape=w_shape)
    d_act = lambda var: 1.0

    x0_temp = np.sum(x0, axis=0)
    x1_temp = np.sum(x1, axis=0)

    y0_temp = np.sum(y0, axis=0)
    y1_temp = np.sum(y1, axis=0)

    if x0_temp.shape[0] < y0_temp.shape[0]:
        x_max = x0_temp.shape[0]
        for i in range(x_max):
            result = (y1_temp - y0_temp) * x1_temp[i] * d_act(y1_temp) + \
                     (x1_temp[i] - x0_temp[i]) * y0_temp * d_act(x1_temp[i])
            grad[i, :] = result
    else:
        y_max = y0_temp.shape[0]
        for j in range(y_max):
            result = (y1_temp[j] - y0_temp[j]) * x1_temp * d_act(y1_temp[j]) + \
                     (x1_temp - x0_temp) * y0_temp[j] * d_act(x1_temp)
            grad[:, j] = result
    return grad


def weight_update_np(x0, x1, y0, y1, w_shape, stride):
    y_shape = y0.shape

    if len(y_shape) == 4:
        if stride == 1:
            grad = weight_update_conv_np_stride_1(x0, x1, y0, y1, w_shape)
        else:
            grad = weight_update_conv_np_stride_n(x0, x1, y0, y1, w_shape, stride)
    elif len(y_shape) == 2:
        grad = weight_update_fc_np(x0, x1, y0, y1, w_shape)
    else:
        raise ValueError('y_shape is {}. Must be 2d or 4d'.format(len(y_shape)))
    return grad


def grad_bias(x, y, stride, d_act, grads, grad_name, prefix):
    if grad_name.find(prefix) != -1:
        w_shape = None
        for name in grads:
            if name.find(prefix) != -1 and name.find('weight') != -1:
                w_shape = grads[name].shape.as_list()
                break
        return grad_bias_x(x0=x[0],
                           x1=x[1],
                           y_shape=y[0].shape,
                           w_shape=w_shape,
                           bias_shape=grads[grad_name].shape.as_list(),
                           stride=stride,
                           d_act=d_act)
    else:
        w_shape = None
        for name in grads:
            if name.find(prefix) == -1 and name.find('weight') != -1:
                w_shape = grads[name].shape.as_list()
                break
        return grad_bias_y(y0=y[0],
                           y1=y[1],
                           w_shape=w_shape,
                           bias_shape=grads[grad_name].shape.as_list(),
                           d_act=d_act)


def grad_bias_x(x0, x1, y_shape, w_shape, bias_shape, stride, d_act):
    grad = np.zeros(bias_shape, np.float32)

    # For convolutional layer
    if len(w_shape) == 4:
        for m in range(w_shape[0]):
            for n in range(w_shape[1]):
                # Get part of x by w(m,n)
                x0_temp = x0[:, m::stride, n::stride, :]
                x0_temp = x0_temp[:, :y_shape[1], :y_shape[2]]

                x1_temp = x1[:, m::stride, n::stride, :]
                x1_temp = x1_temp[:, :y_shape[1], :y_shape[2]]

                # We will get result with shape=(batch_size, y_height, y_width, x_maps_count). 'x_maps_count' it is 'q'
                result = (x1_temp - x0_temp) * d_act(x1_temp)
                grad[:] += np.sum(result)
        grad /= y_shape[1] * y_shape[2]
    elif len(w_shape) == 2:
        result = (x1 - x0) * d_act(x1)
        grad[:] = np.sum(result, axis=0)
    else:
        raise ValueError('Weights shape is {}. Must be 2d or 4d'.format(w_shape))

    return grad


def grad_bias_y(y0, y1, w_shape, bias_shape, d_act):
    grad = np.zeros(bias_shape, dtype=np.float32)
    y_shape = y0.shape

    # We will get result with shape=(batch_size, y_height, y_width, y_maps_count). 'y_maps_count' it is 'k'
    result = (y1 - y0) * d_act(y1)

    if len(w_shape) == 4:
        grad[:] += np.sum(result)
        grad /= y_shape[1] * y_shape[2]
    elif len(w_shape) == 2:
        grad[:] = np.sum(result, axis=0)
    else:
        raise ValueError('Weights shape is {}. Must be 2d or 4d'.format(w_shape))

    return grad


def array_norm(array, norm_value):
    for i in range(len(array)):
        array[i] = array[i] / norm_value


class TensorShape:
    def __init__(self, shape):
        self.shape = shape

    def as_list(self):
        return self.shape


class Shape:
    def __init__(self, shape):
        self.shape = TensorShape(shape)


def main():
    x_size = 27
    y_size = 24
    x_channel = 3
    batch_size = 1
    y_output = 6
    formulas = Formulas.hinton
    act_name = 'relu'

    d_act = d_act_dic[act_name] if formulas == Formulas.golovko else lambda var: 1.0

    w_shape = [2, 2, x_channel, y_output]

    x_side = int(sqrt(x_size / x_channel))
    y_side = int(sqrt(y_size / y_output))

    x0 = np.arange(x_size, dtype=np.float32)
    x1 = np.arange(x_size, dtype=np.float32)

    array_norm(x0, norm_value=len(x0) + 1.0)
    array_norm(x1, norm_value=2 * len(x1) + 1.0)

    x0 = x0.reshape(batch_size, x_side, x_side, x_channel)
    x1 = x1.reshape(batch_size, x_side, x_side, x_channel)

    y0 = np.arange(y_size, dtype=np.float32)
    y1 = np.arange(y_size, dtype=np.float32)

    array_norm(y0, norm_value=len(y0) + 1.0)
    array_norm(y1, norm_value=2 * len(y1) + 1.0)

    y0 = y0.reshape(batch_size, y_side, y_side, y_output)
    y1 = y1.reshape(batch_size, y_side, y_side, y_output)

    # x_size = 5
    # y_size = 3
    # batch_size = 1
    #
    # w_shape = [x_size, y_size]
    #
    # x0 = np.arange(x_size, dtype=np.float32)
    # x1 = np.arange(x_size, dtype=np.float32)
    #
    # y0 = np.arange(y_size, dtype=np.float32)
    # y1 = np.arange(y_size, dtype=np.float32)
    #
    # array_norm(x0, norm_value=len(x0) + 1.0)
    # array_norm(x1, norm_value=2 * len(x1) + 1.0)
    #
    # array_norm(y0, norm_value=len(y0) + 1.0)
    # array_norm(y1, norm_value=2 * len(y1) + 1.0)
    #
    # x0.resize(batch_size, x_size)
    # x1.resize(batch_size, x_size)
    #
    # y0.resize(batch_size, y_size)
    # y1.resize(batch_size, y_size)

    # x0 = np.array([0.1, 0.2, 0.3])
    # x1 = np.array([0.11, 0.22, 0.33])
    #
    # y0 = np.array([0.2, 0.4, 0.6])
    # y1 = np.array([0.22, 0.44, 0.66])
    #
    # x0 = np.array([[0.5, 0.6],
    #                [0.4, 0.7]])
    #
    # x1 = np.array([[0.0, 0.18],
    #                [0.551, 0.324]])
    #
    # w1 = np.array([[-0.5, 0.4],
    #                [0.6, 0.5]])
    #
    # y0 = 0.585
    # y1 = 0.57
    #
    # x0 = np.resize(x0, new_shape=(1, 2, 2, 1))
    # x1 = np.resize(x1, new_shape=(1, 2, 2, 1))
    # y0 = np.resize(y0, new_shape=(1, 1, 1, 2))
    # y1 = np.resize(y1, new_shape=(1, 1, 1, 2))
    # w_shape = [2, 2, 1, 2]

    conv1_weights = Shape(w_shape)
    conv1_biases = Shape([conv1_weights.shape.as_list()[-1]])

    input_recovery_weights = Shape(w_shape)
    input_recovery_biases = Shape([input_recovery_weights.shape.as_list()[-2]])

    grads = {'Model/conv1/weights:0': conv1_weights,
             'Model/conv1/biases:0': conv1_biases,
             'Model/input_recovery/weights:0': input_recovery_weights,
             'Model/input_recovery/biases:0': input_recovery_biases}

    start_normal = timer()
    # grad_normal = weights_update(x0, x1, y0, y1, w_shape, stride=1)
    end_normal = timer() - start_normal

    start_np = timer()
    # grad_np = weight_update_np(x0, x1, y0, y1, w_shape, stride=1)
    grad_bias([x0, x1],
              [y0, y1],
              stride=1,
              d_act=d_act,
              grads=grads,
              grad_name='Model/input_recovery/biases:0',
              prefix='recovery')
    grad_bias([x0, x1],
              [y0, y1],
              stride=1,
              d_act=d_act,
              grads=grads,
              grad_name='Model/conv1/biases:0',
              prefix='recovery')
    end_np = timer() - start_np

    # print(grad_normal)
    # print(grad_np)
    print('function_normal %f seconds: ' % end_normal)
    print('function_np %f seconds: ' % end_np)

    print('normal / np:  %f' % (end_normal / end_np))
    # print("grad_normal == grad_np: {}".format(np.equal(grad_normal, grad_np)))


if __name__ == '__main__':
    main()
