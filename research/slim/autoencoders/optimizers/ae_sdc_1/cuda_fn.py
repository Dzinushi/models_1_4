# from numba import jit, cuda, vectorize
import numpy as np


def multiply(array_1, array_2):
    return np.append(array_1, np.zeros(0 if (len(array_2) - len(array_1)) < 0
                                       else (len(array_2) - len(array_1)))) * \
           np.append(array_2, np.zeros(0 if (len(array_1) - len(array_2)) < 0
                                       else (len(array_1) - len(array_2))))


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
                           d_act=d_act)


def grad_bias_x(x0, x1, y_shape, w_shape, stride, d_act):
    grad = np.zeros(w_shape[-1], np.float32)

    # For convolutional layer
    if len(w_shape) == 4:
        for m in range(w_shape[0]):
            for n in range(w_shape[1]):
                for c in range(w_shape[2]):
                    # Get part of x by w(m,n)
                    x0_temp = x0[:, m::stride, n::stride, c]
                    x0_temp = x0_temp[:, :y_shape[1], :y_shape[2]]

                    x1_temp = x1[:, m::stride, n::stride, c]
                    x1_temp = x1_temp[:, :y_shape[1], :y_shape[2]]

                    # Summary x by c (number of channel)
                    x0_temp = np.sum(x0_temp, axis=0)
                    x1_temp = np.sum(x1_temp, axis=0)

                    result = (x1_temp - x0_temp) * d_act(x1_temp)

                    grad[:] += np.sum(result, axis=(0, 1))
    elif len(w_shape) == 2:
        x0_temp = np.sum(x0, axis=0)
        x1_temp = np.sum(x1, axis=0)
        grad[:] = (x1_temp - x0_temp) * d_act(x1_temp)
    else:
        raise ValueError('Weights shape is {}. Must be 2d or 4d'.format(w_shape))

    return grad
    # # Convolution layers
    # if len(x_shape) == 4:
    #     for q in range(x_shape[0]):
    #         t = []
    #         for c in range(x_shape[3]):
    #             for m in range(w_shape[0]):
    #                 for n in range(w_shape[1]):
    #                     for i in range(y_shape[0]):
    #                         for j in range(y_shape[1]):
    #                             t.append(xe[q][i * stride + m][j * stride + n][c] * \
    #                                      d_act(x[1][q][i * stride + m][j * stride + n][c]))
    #         grad_value[q] = np.sum(t)
    #
    # # Full connections layers: k = q = batch_size
    # else:
    #     for k in range(x_shape[0]):
    #         for i in range(x_shape[1]):
    #             grad_value[i] = xe[k][i] * d_act(x[1][k][i])
    #
    # return grad_value


def grad_bias_y(y0, y1, w_shape, d_act):

    grad = np.zeros(w_shape[-1], dtype=np.float32)
    y0_temp = np.sum(y0, axis=0)
    y1_temp = np.sum(y1, axis=0)

    if len(w_shape) == 4:
        result = (y1_temp - y0_temp) * d_act(y1_temp)
        grad[:] = np.sum(result, axis=(0, 1))
    elif len(w_shape) == 2:
        grad[:] = (y1_temp - y0_temp) * d_act(y1_temp)
    else:
        raise ValueError('Weights shape is {}. Must be 2d or 4d'.format(w_shape))

    return grad

    # Convolution layers
    # if len(y_shape) == 4:
    #     for k in range(y_shape[3]):
    #         t = []
    #         for c in range(y_shape[0]):
    #             for i in range(y_shape[1]):
    #                 for j in range(y_shape[2]):
    #                     t.append(ye[c][i][j][k] * \
    #                              d_act(y[1][c][i][j][k]))
    #         grad_value[k] = np.sum(t)
    #
    # # Full connections layers: k = q = batch_size
    # else:
    #     for k in range(y_shape[0]):
    #         for j in range(y_shape[1]):
    #             grad_value[j] = ye[k][j] * d_act(y[1][k][j])


def grad_weight(x, y, w_shape, stride, d_act):
    y_shape = y[0].shape

    if len(y_shape) == 4:
        if stride == 1:
            grad = weight_update_conv_stride_1(x[0], x[1], y[0], y[1], w_shape, d_act)
        else:
            grad = weight_update_conv_stride_n(x[0], x[1], y[0], y[1], w_shape, stride, d_act)
    elif len(y_shape) == 2:
        grad = weight_update_fc(x[0], x[1], y[0], y[1], w_shape, d_act)
    else:
        raise ValueError('y_shape is {}. Must be 2d or 4d'.format(len(y_shape)))
    return grad


def weight_update_conv_stride_1(x0, x1, y0, y1, w_shape, d_act):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

    for m in range(w_shape[0]):
        for n in range(w_shape[1]):
            for c in range(w_shape[2]):
                # Get part of x by w(m,n)
                x0_temp = x0[:, m:y_shape[1] + m, n:y_shape[2] + n, c]
                x1_temp = x1[:, m:y_shape[1] + m, n:y_shape[2] + n, c]

                # Summary x by c (number of channel)
                x0_temp = np.sum(x0_temp, axis=0)
                x1_temp = np.sum(x1_temp, axis=0)

                y0_temp = np.sum(y0, axis=0)
                y1_temp = np.sum(y1, axis=0)

                # Resize x
                x0_temp = np.resize(x0_temp, new_shape=(x0_temp.shape[0], x0_temp.shape[1], 1))
                x1_temp = np.resize(x1_temp, new_shape=x0_temp.shape)

                result = (x1_temp - x0_temp) * y0_temp * d_act(x1_temp) + \
                         (y1_temp - y0_temp) * x1_temp * d_act(y1_temp)

                grad[m, n, c, :] = np.sum(result, axis=(0, 1))
    return grad


def weight_update_conv_stride_n(x0, x1, y0, y1, w_shape, stride, d_act):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

    for m in range(w_shape[0]):
        for n in range(w_shape[1]):
            for c in range(w_shape[2]):
                # Get part of x by w(m,n)
                x0_temp = x0[:, m::stride, n::stride, c]
                x0_temp = x0_temp[:, :y_shape[1], :y_shape[2]]

                x1_temp = x1[:, m::stride, n::stride, c]
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

                grad[m, n, c, :] = np.sum(result, axis=(0, 1))
    return grad


def weight_update_fc(x0, x1, y0, y1, w_shape, d_act):
    grad = np.zeros(shape=w_shape)

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
