import numpy as np


# @vectorize(["float32(float32)"], target='cuda')
def d_relu_cuda(y):
    y_d_relu = np.copy(y)
    y_d_relu[y_d_relu > 0] = 1.0
    y_d_relu[y_d_relu < 0] = 0.0
    return y_d_relu


# @vectorize(["float32(float32)"], target='cuda')
def d_leakyrelu_cuda(y):
    y_d_leakyrelu = np.copy(y)
    y_d_leakyrelu[y_d_leakyrelu > 0] = 1.0
    y_d_leakyrelu[y_d_leakyrelu <= 0] = 0.2
    return y_d_leakyrelu


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


# Производная активации
# TODO: old, not using in fast cpu version
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
