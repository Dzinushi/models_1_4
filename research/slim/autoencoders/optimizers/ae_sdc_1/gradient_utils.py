import numpy as np

# w_shape=(HWQK)
# x_shape=(NHWQ)
# y_shape=(NHWK)


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

        # Division by i * j * q * batch_size
        grad /= y_shape[1] * y_shape[2] * x0.shape[3] * x0.shape[0]

    elif len(w_shape) == 2:
        result = (x1 - x0) * d_act(x1)
        grad[:] = np.sum(result, axis=0)

        # Division by batch_size
        grad /= y_shape[0]
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

        # Division by i * j * k * batch_size
        grad /= y_shape[1] * y_shape[2] * y_shape[3] * y_shape[0]
    elif len(w_shape) == 2:
        grad[:] = np.sum(result, axis=0)

        # Division by batch_size
        grad /= y_shape[0]
    else:
        raise ValueError('Weights shape is {}. Must be 2d or 4d'.format(w_shape))

    return grad


def grad_w_conv_stride_1(x0, x1, y0, y1, w_shape, d_act):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

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
                x0_temp = np.reshape(x0_temp, newshape=(x0_temp.shape[0], x0_temp.shape[1], 1))
                x1_temp = np.reshape(x1_temp, newshape=x0_temp.shape)

                result = (x1_temp - x0_temp) * y0_temp * d_act(x1_temp) + \
                         (y1_temp - y0_temp) * x1_temp * d_act(y1_temp)

                grad[m, n, q, :] = np.sum(result, axis=(0, 1))

    # Division by i * j * batch_size
    grad /= y_shape[1] * y_shape[2] * y_shape[0]
    return grad


def grad_w_conv_stride_n(x0, x1, y0, y1, w_shape, stride, d_act):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

    for m in range(w_shape[0]):
        for n in range(w_shape[1]):
            for q in range(w_shape[2]):
                # Get part of x by w(m,n)
                x0_temp = x0[:, m::stride, n::stride, q]
                x0_temp = x0_temp[:, :y_shape[1], :y_shape[2]]

                x1_temp = x1[:, m::stride, n::stride, q]
                x1_temp = x1_temp[:, :y_shape[1], :y_shape[2]]

                # Summary x by batch size
                x0_temp = np.sum(x0_temp, axis=0)
                x1_temp = np.sum(x1_temp, axis=0)

                y0_temp = np.sum(y0, axis=0)
                y1_temp = np.sum(y1, axis=0)

                # Resize x
                x0_temp = np.reshape(x0_temp, newshape=(x0_temp.shape[0], x0_temp.shape[1], 1))
                x1_temp = np.reshape(x1_temp, newshape=x0_temp.shape)

                result = (x1_temp - x0_temp) * y0_temp * d_act(x1_temp) + \
                         (y1_temp - y0_temp) * x1_temp * d_act(y1_temp)

                grad[m, n, q, :] = np.sum(result, axis=(0, 1))

    # Division by i * j * batch_size
    grad /= y_shape[1] * y_shape[2] * y_shape[0]
    return grad


def grad_w_fc(x0, x1, y0, y1, w_shape, grad_name, prefix, d_act):
    grad = np.zeros(shape=w_shape)

    x0_temp = np.sum(x0, axis=0)
    x1_temp = np.sum(x1, axis=0)

    y0_temp = np.sum(y0, axis=0)
    y1_temp = np.sum(y1, axis=0)

    if x0_temp.shape[0] < y0_temp.shape[0]:
        x_max = x0_temp.shape[0]

        # Convolution layer: w_shape = (x_temp.shape[0], y_temp.shape[0])
        if grad_name.find(prefix) == -1:
            for i in range(x_max):
                result = (y1_temp - y0_temp) * x1_temp[i] * d_act(y1_temp) + \
                         (x1_temp[i] - x0_temp[i]) * y0_temp * d_act(x1_temp[i])
                grad[i, :] = result

        # Convolution layer: w_shape = (y_temp.shape[0], x_temp.shape[0])
        else:
            for i in range(x_max):
                result = (y1_temp - y0_temp) * x1_temp[i] * d_act(y1_temp) + \
                         (x1_temp[i] - x0_temp[i]) * y0_temp * d_act(x1_temp[i])
                grad[:, i] = result
    else:
        y_max = y0_temp.shape[0]
        if grad_name.find(prefix) == -1:
            for j in range(y_max):
                result = (y1_temp[j] - y0_temp[j]) * x1_temp * d_act(y1_temp[j]) + \
                         (x1_temp - x0_temp) * y0_temp[j] * d_act(x1_temp)
                grad[:, j] = result
        else:
            for j in range(y_max):
                result = (y1_temp[j] - y0_temp[j]) * x1_temp * d_act(y1_temp[j]) + \
                         (x1_temp - x0_temp) * y0_temp[j] * d_act(x1_temp)
                grad[j, :] = result

    # Division by batch_size
    grad /= y0.shape[0]
    return grad
