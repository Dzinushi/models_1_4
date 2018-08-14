import numpy as np
from math import e, pow

# 0-я итерация

input_sdc_0 = np.array([
    [
        0.5,
        0.6000000238418579
    ],
    [
        0.4000000059604645,
        0.699999988079071
    ],
])
input_sdc_1 = np.array([
    [
        0.0,
        0.18060001730918884
    ],
    [
        0.5514000654220581,
        0.3237000107765198
    ],
])
output_sdc_0 = np.array([
    0.5850000381469727,
    0.28200000524520874
])
output_sdc_1 = np.array([
    0.5699300765991211,
    0.3842300474643707
])
weights_output_sdc_0 = np.array([
    [
        [
            -0.5,
            0.10000000149011612
        ],
        [
            0.4000000059604645,
            -0.20000000298023224
        ],
    ],
    [
        [
            0.6000000238418579,
            0.699999988079071
        ],
        [
            0.5,
            0.10000000149011612
        ],
    ],
])
biases_output_sdc_0 = np.array([
    0.004999999888241291,
    0.0020000000949949026
])
weights_input_sdc_1 = np.array([
    [
        [
            -0.5,
            0.10000000149011612
        ],
        [
            0.4000000059604645,
            -0.20000000298023224
        ],
    ],
    [
        [
            0.6000000238418579,
            0.699999988079071
        ],
        [
            0.5,
            0.10000000149011612
        ],
    ],
])
biases_input_sdc_1 = np.array([
    0.003000000026077032
])


# Weights shape: HWCN
# Biases shape: N
# Input shape: NHWC
# Output shape: HWCN


# Вычисление взвешенной суммы для y0, y1 ... yk:
def reduce_sum_y(x, w, bias):
    sum = np.zeros(w.shape[2])  # k
    for k in range(w.shape[2]):
        s = []
        for m in range(w.shape[0]):
            for n in range(w.shape[1]):
                s.append(w[m][n][k] * x[m][n])
        sum[k] = np.sum(s) + bias[k]
    return sum


# Вычисление взвешенной суммы для x0, x1 ... xq
def reduce_sum_x(y, w, bias):
    sum = np.zeros(shape=(w.shape[0], w.shape[1]))
    for m in range(w.shape[0]):
        for n in range(w.shape[1]):
            s = []
            for k in range(w.shape[2]):
                s.append(w[m][n][k] * y[k])
            sum[m][n] = np.sum(s) + bias
    return sum


def relu(s):
    return s if s > 0.0 else 0.0


def leaky_relu(s, const=0.2):
    return s if s > 0.0 else const * s


def sigm(s):
    return 1.0 / (1.0 + pow(e, -s))


def tanh(s):
    # return (pow(e, s) - pow(e, -s)) / (pow(e, s) + pow(e, -s))
    return 2.0 / (1.0 + pow(e, -2.0 * s)) - 1.0


# Применение активации
def activation_y(s, name):
    y = []
    for var in s:
        if name == 'relu':
            y.append(relu(var))
        elif name == 'leaky_relu':
            y.append(leaky_relu(var))
        elif name == 'sigm':
            y.append(sigm(var))
        elif name == 'tanh':
            y.append(tanh(var))
        else:
            raise ValueError('Unrecognized activation: ', name)
    return y


def activation_x(s, name):
    x = np.zeros(s.shape)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if name == 'relu':
                x[i][j] = relu(s[i][j])
            elif name == 'leaky_relu':
                x[i][j] = leaky_relu(s[i][j])
            elif name == 'sigm':
                x[i][j] = sigm(s[i][j])
            elif name == 'tanh':
                x[i][j] = tanh(s[i][j])
    return x


# Производная активации
def d_activation(s, name):
    result = None
    if name == 'relu':
        result = 1.0 if s > 0 else 0.0
    elif name == 'leaky_relu':
        result = 1.0 if s > 0 else 0.2
    elif name == 'sigm':
        result = sigm(s) * (1.0 - sigm(s))
    elif name == 'tanh':
        result = 1.0 - pow(tanh(s), 2)
    return result


# Обновление весов (возможно стоит добавить деление градиента на i*j)
def weights_update(w, x0, x1, y0, y1, sx1, sy1, alpha=0.01, name='relu'):
    w_new = np.zeros(w.shape)
    grad = np.zeros(w.shape)
    for k in range(w.shape[2]):
        for m in range(w.shape[0]):
            for n in range(w.shape[1]):
                grad[m][n][k] = (y1[k] - y0[k]) * d_activation(sy1[k], name) * x1[m][n] + \
                                (x1[m][n] - x0[m][n]) * d_activation(sx1[m][n], name) * y0[k]
                w_new[m][n][k] = w[m][n][k] - alpha * grad[m][n][k]
    return w_new, grad


def weights_update_hinton(w, x0, x1, y0, y1, alpha=0.01):
    w_new = np.zeros(w.shape)
    grad = np.zeros(w.shape)
    for k in range(w.shape[2]):
        for m in range(w.shape[0]):
            for n in range(w.shape[1]):
                grad[m][n][k] = (y1[k] - y0[k]) * x1[m][n] + \
                                (x1[m][n] - x0[m][n]) * y0[k]
                w_new[m][n][k] = w[m][n][k] - alpha * grad[m][n][k]
    return w_new, grad


# Обновление порогов
def biases_update_y(biases, y0, y1, sy1, alpha=0.01, name='relu'):
    biases_new = np.zeros(biases.shape)
    grad = np.zeros(biases.shape)
    for k in range(biases.shape[0]):
        grad[k] = (y1[k] - y0[k]) * d_activation(sy1[k], name)
        biases_new[k] = biases[k] - alpha * grad[k]
    return biases_new, grad


def biases_update_x(biases, x0, x1, sx1, alpha=0.01, name='relu'):
    s = []
    for m in range(x0.shape[0]):
        for n in range(x0.shape[1]):
            s.append((x1[m][n] - x0[m][n]) * d_activation(sx1[m][n], name))
    grad = np.sum(s)
    biases_new = biases - alpha * grad
    return biases_new, grad


def biases_update_x_hinton(biases, x0, x1, alpha=0.01):
    s = []
    for m in range(x0.shape[0]):
        for n in range(x0.shape[1]):
            s.append((x1[m][n] - x0[m][n]))
    grad = np.sum(s)
    biases_new = biases - alpha * grad
    return biases_new, grad


def biases_update_y_hinton(biases, y0, y1, alpha=0.01):
    biases_new = np.zeros(biases.shape)
    grad = np.zeros(biases.shape)
    for k in range(biases.shape[0]):
        grad[k] = (y1[k] - y0[k])
        biases_new[k] = biases[k] - alpha * grad[k]
    return biases_new, grad


x0 = input_sdc_0
x1 = input_sdc_1
y0 = output_sdc_0
y1 = output_sdc_1
wy0 = weights_output_sdc_0
wx1 = weights_input_sdc_1
biases_y0 = biases_output_sdc_0
biases_x1 = biases_input_sdc_1

max_step = 1000
log_data = None
activation_name = 'sigm'
update = 'golovko'
# update = 'hinton'

for step in range(max_step):

    sy0 = reduce_sum_y(x0, wy0, biases_y0)
    y0 = activation_y(sy0, activation_name)

    sx1 = reduce_sum_x(y0, wx1, biases_x1)
    x1 = activation_x(sx1, activation_name)

    sy1 = reduce_sum_y(x1, wy0, biases_y0)
    y1 = activation_y(sy1, activation_name)

    # Golovko method
    if update == 'golovko':
        wy0_1, grad_wy0_1 = weights_update(w=wy0,
                                           x0=x0,
                                           x1=x1,
                                           y0=y0,
                                           y1=y1,
                                           sx1=sx1,
                                           sy1=sy1,
                                           name=activation_name)
        wx1_1, grad_wx1_1 = weights_update(w=wx1,
                                           x0=x0,
                                           x1=x1,
                                           y0=y0,
                                           y1=y1,
                                           sx1=sx1,
                                           sy1=sy1,
                                           name=activation_name)
        biases_y0_1, grad_biases_y0_1 = biases_update_y(biases=biases_y0,
                                                        y0=y0,
                                                        y1=y1,
                                                        sy1=sy1,
                                                        name=activation_name)
        biases_x1_1, grad_biases_x1_1 = biases_update_x(biases=biases_x1,
                                                        x0=x0,
                                                        x1=x1,
                                                        sx1=sx1,
                                                        name=activation_name)
    # Hinton method
    elif update == 'hinton':
        wy0_1, grad_wy0_1 = weights_update_hinton(w=wy0,
                                                  x0=x0,
                                                  x1=x1,
                                                  y0=y0,
                                                  y1=y1)
        wx1_1, grad_wx1_1 = weights_update_hinton(w=wx1,
                                                  x0=x0,
                                                  x1=x1,
                                                  y0=y0,
                                                  y1=y1)
        biases_y0_1, grad_biases_y0_1 = biases_update_y_hinton(biases=biases_y0,
                                                               y0=y0,
                                                               y1=y1)
        biases_x1_1, grad_biases_x1_1 = biases_update_x_hinton(biases=biases_x1,
                                                               x0=x0,
                                                               x1=x1)
    else:
        raise ValueError('Failed update by name: ', update)

    loss_y = np.sum(np.square(np.subtract(y1, y0)))
    loss_x = np.sum(np.square(np.subtract(x1, x0)))
    loss = (loss_y + loss_x) / 2.

    if log_data is not None and (step + 1) % log_data == 0:
        print('Sy0: {}'.format(sy0))
        print('Sx1: {}'.format(sx1))
        print('Sy1: {}'.format(sy1))

        print('y0: {}'.format(y0))
        print('x1: {}'.format(x1))
        print('y1: {}'.format(y1))

        print('Wy0(t+1), t=0:\n{}\n'.format(wy0_1))
        print('Wx1(t+1), t=0:\n{}\n'.format(wx1_1))
        print('Gradient Wy0(t+1), t=0:\n{}\n'.format(grad_wy0_1))
        print('Gradient Wx1(t+1), t=0:\n{}\n'.format(grad_wx1_1))

        print('Ty0(t+1), t=0:{}'.format(biases_y0_1))
        print('Tx1(t+1), t=0:{}'.format(biases_x1_1))
        print('Gradient Ty0(t+1), t=0:{}'.format(grad_biases_y0_1))
        print('Gradient Tx1(t+1), t=0:{}\n'.format(grad_biases_x1_1))

    print('{}) Loss: {}'.format(step + 1, loss))

    wy0 = wy0_1
    wx1 = wx1_1
    biases_y0 = biases_y0_1
    biases_x1 = biases_x1_1
