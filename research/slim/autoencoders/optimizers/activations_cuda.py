import numpy as np
from time import time
from numba import cuda, double, void, jit
# from autoencoders.optimizers.ae_sdc_1.gradient import grad_w_conv_stride_1


@cuda.jit(double[:, :, :, :](double[:, :, :, :]), device=True)
def d_relu_cuda(y):
    result = np.copy(y)
    result[y > 0] = 1.0
    result[y < 0] = 0.0
    return result


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


def grad_w_conv_stride_1(x0, x1, y0, y1, w_shape):
    grad = np.zeros(shape=w_shape)
    y_shape = y0.shape

    d_act = d_relu_cuda

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


@jit(double[:,:,:,:](double[:,:,:,:], double[:,:,:,:]))
def multiply_gpu(a, b):
    return a * b


def multiply_cpu(a, b):
    return a * b


def main():

    # Test grad_w_conv_stride_1 with cuda

    x0 = np.random.rand(1, 28, 28, 3)
    x1 = np.random.rand(1, 28, 28, 3)
    y0 = np.random.rand(1, 24, 24, 32)
    y1 = np.random.rand(1, 24, 24, 32)
    w_shape = (5, 5, 3, 32)

    # grad_w_conv_stride_1_gpu = jit(double[:,:,:,:](double[:,:,:,:],
    #                                                double[:,:,:,:],
    #                                                double[:,:,:,:],
    #                                                double[:,:,:,:],
    #                                                double[:,:,:,:]))(grad_w_conv_stride_1)

    time_start = time()
    d_relu_cuda(x0)
    # grad_w_conv_stride_1(x0, x1, y0, y1, w_shape)
    # multiply_cpu(x0, x1)
    time_end_cpu = time() - time_start

    time_start = time()
    d_leakyrelu_cuda(x0)
    # grad_w_conv_stride_1_gpu(x0, x1, y0, y1, w_shape)
    # multiply_gpu(x0, x1)
    time_end_gpu = time() - time_start

    print('Time cpu: %f' % time_end_cpu)
    print('Time gpu: %f' % time_end_gpu)
    print('gpu/cpu = %f' % (time_end_gpu / time_end_cpu))


if __name__ == '__main__':
    main()
