import numpy as np
from time import time
from numba import cuda, double, void, jit


# from autoencoders.optimizers.ae_sdc_1.gradient import grad_w_conv_stride_1


# @cuda.jit(double[:, :, :, :](double[:, :, :, :]), device=True)
def d_relu_np(y):
    result = np.copy(y)
    result[y > 0] = 1.0
    result[y < 0] = 0.0
    return result


@cuda.jit(void(double[:]))
def d_relu_cuda(y):
    pos = cuda.grid(1)
    if y[pos] > 0.0:
        y[pos] = 1.0
    else:
        y[pos] = 0.0


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


d_act_dic = {'relu': d_relu_np,
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


@cuda.jit(void(double[:], double[:], double[:], double[:], double[:], double[:]))
def grad_w_conv_stride_1(x0, x1, y0, y1, x_shape, y_shape, w_shape, grad):
    """
    x[batch][height][width][map] -> x[batch * height * width * map]
    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :param x_shape: 1d array contained shape of x0 and x1 arrays
    :param y_shape: 1d array contained shape of x0 and x1 arrays
    :param w_shape: 1d array contained shape of weights
    :param grad:
    :return:
    """
    y_shape = y0.shape

    d_act = d_relu_np

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


@cuda.jit(void(double[:]))
def increment_by_one_cuda(array):
    pos = cuda.grid(1)
    if pos < array.size:
        array[pos] += 1


def increment_by_one(array):
    array += 1
    return array


# Running activation functions in gpu mode
def run_in_gpu_mode(array, func, threadperblock=32):
    # Original shape
    array_shape = array.shape

    # Reshape to 1d array
    array = array.reshape(np.prod(array_shape))

    # GPU options
    blockpergrid = (array.size + threadperblock - 1) // threadperblock

    # Send array to gpu memory
    array_in_gpu = cuda.to_device(array)

    # Run function in gpu
    func[blockpergrid, threadperblock](array_in_gpu)

    # Reshape result to original shape and return to user
    return array_in_gpu.copy_to_host().reshape(array_shape)


def time_fn(func, array, func_act=None):
    time_start = time()
    if func_act is not None:
        result = func(array, func_act)
    else:
        result = func(array)
    time_end_np = time() - time_start
    return result, time_end_np


def run_relu():
    array = np.random.rand(1, 28, 28, 3)
    array_cpu, time_end_cpu = time_fn(d_relu_np, array)
    array_gpu, time_end_gpu = time_fn(run_in_gpu_mode, array, d_relu_cuda)
    print('Time cpu: %f' % time_end_cpu)
    print('Time gpu: %f' % time_end_gpu)
    print('gpu/cpu = %f' % (time_end_gpu / time_end_cpu))


def main():
    array_1 = np.zeros(10)

    # CPU time
    time_start = time()
    array_cpu_1 = increment_by_one(array_1)
    time_end_cpu = time() - time_start

    d_array_1 = cuda.to_device(array_1)

    # CUDA time
    time_start = time()
    run_in_gpu_mode(d_array_1, increment_by_one_cuda)
    time_end_gpu = time() - time_start

    array_gpu_1 = d_array_1.copy_to_host()

    print('Array cpu: {}'.format(array_cpu_1[0:5]))
    print('Array gpu: {}'.format(array_gpu_1[0:5]))
    print('Time cpu: %f' % time_end_cpu)
    print('Time gpu: %f' % time_end_gpu)
    print('gpu/cpu = %f' % (time_end_gpu / time_end_cpu))


if __name__ == '__main__':
    run_relu()
