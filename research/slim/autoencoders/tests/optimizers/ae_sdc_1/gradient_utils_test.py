import unittest
import numpy as np
from research.slim.autoencoders.optimizers.ae_sdc_1.gradient_utils import grad_w_conv_stride_1, grad_w_conv_stride_n, grad_bias_x, grad_w_fc
from research.slim.autoencoders.optimizers.activations import d_relu_cuda, d_leakyrelu_cuda, d_sigmoid_cuda, d_tanh_cuda

np.random.seed(7)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CustomGradientSDC1Arguments_stride_1(metaclass=Singleton):
    def __init__(self):
        self._x0 = round_4d(np.random.rand(1, 3, 3, 2), accuracy=1)
        self._x1 = round_4d(np.random.rand(1, 3, 3, 2), accuracy=1)
        self._y0 = round_4d(np.random.rand(1, 2, 2, 3), accuracy=1)
        self._y1 = round_4d(np.random.rand(1, 2, 2, 3), accuracy=1)


class CustomGradientSDC1Arguments_stride_n(metaclass=Singleton):
    def __init__(self):
        self._x0 = round_4d(np.random.rand(1, 4, 4, 2), accuracy=1)
        self._x1 = round_4d(np.random.rand(1, 4, 4, 2), accuracy=1)
        self._y0 = round_4d(np.random.rand(1, 2, 2, 3), accuracy=1)
        self._y1 = round_4d(np.random.rand(1, 2, 2, 3), accuracy=1)


class CustomGradientSDC1Arguments_fc(metaclass=Singleton):
    def __init__(self):
        self._x0 = round_2d(np.random.rand(1, 5), accuracy=1)
        self._x1 = round_2d(np.random.rand(1, 5), accuracy=1)
        self._y0 = round_2d(np.random.rand(1, 3), accuracy=1)
        self._y1 = round_2d(np.random.rand(1, 3), accuracy=1)


def array_normilize_one(array):
    max_value = array.max()
    for i in range(len(array)):
        array[i] = array[i] / max_value


def round_2d(array, accuracy=3):
    array_r = np.zeros(array.shape)
    for batch in range(array_r.shape[0]):
        for i in range(array_r.shape[1]):
            array_r[batch][i] = round(array[batch][i], accuracy)
    return array_r


def round_4d(array, accuracy=3):
    array_r = np.zeros(array.shape)
    for h in range(array.shape[0]):
        for w in range(array.shape[1]):
            for q in range(array.shape[2]):
                for k in range(array.shape[3]):
                    array_r[h][w][q][k] = round(array[h][w][q][k], accuracy)
    return array_r


class TestCustomGradientSDC1(unittest.TestCase):

    """Test function 'grad_w_conv_stride_1()'
    Using Hinton and Golovko approaches. Testing 'relu', 'leakyrelu', 'sigmoid' and 'tanh' function activations"""

    # test hinton method
    def test_grad_w_conv_stride_1_hinton(self):
        args = CustomGradientSDC1Arguments_stride_1()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        d_act = lambda var: 1.0

        accuracy = 4

        grad = grad_w_conv_stride_1(x0, x1, y0, y1, w_shape, d_act)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0][0][0][0] = 0.0925
        grad_manual[0][1][0][0] = -0.1075000
        grad_manual[1][0][0][0] = -0.0600000
        grad_manual[1][1][0][0] = 0.0050000

        grad_manual[0][0][0][1] = 0.172500
        grad_manual[0][1][0][1] = -0.032500
        grad_manual[1][0][0][1] = 0.137500
        grad_manual[1][1][0][1] = 0.082500

        grad_manual[0][0][0][2] = 0.0900000
        grad_manual[0][1][0][2] = -0.1350000
        grad_manual[1][0][0][2] = 0.0050000
        grad_manual[1][1][0][2] = 0.0200000

        grad_manual[0][0][1][0] = -0.0675000
        grad_manual[0][1][1][0] = -0.1375000
        grad_manual[1][0][1][0] = -0.0100000
        grad_manual[1][1][1][0] = -0.1250000

        grad_manual[0][0][1][1] = 0.042500
        grad_manual[0][1][1][1] = 0.067500
        grad_manual[1][0][1][1] = 0.030000
        grad_manual[1][1][1][1] = 0.037500

        grad_manual[0][0][1][2] = -0.047500
        grad_manual[0][1][1][2] = -0.127500
        grad_manual[1][0][1][2] = -0.070000
        grad_manual[1][1][1][2] = -0.085000

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'relu'
    def test_grad_w_conv_stride_1_golovko_relu(self):
        args = CustomGradientSDC1Arguments_stride_1()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 4

        grad = grad_w_conv_stride_1(x0, x1, y0, y1, w_shape, d_relu_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0][0][0][0] = 0.0925000
        grad_manual[0][1][0][0] = -0.1075000
        grad_manual[1][0][0][0] = -0.0600000
        grad_manual[1][1][0][0] = 0.0050000

        grad_manual[0][0][0][1] = 0.262500
        grad_manual[0][1][0][1] = 0.017500
        grad_manual[1][0][0][1] = 0.157500
        grad_manual[1][1][0][1] = 0.122500

        grad_manual[0][0][0][2] = 0.0900000
        grad_manual[0][1][0][2] = -0.1350000
        grad_manual[1][0][0][2] = 0.0050000
        grad_manual[1][1][0][2] = 0.0200000

        grad_manual[0][0][1][0] = 0.0925000
        grad_manual[0][1][1][0] = -0.1375000
        grad_manual[1][0][1][0] = -0.0100000
        grad_manual[1][1][1][0] = -0.1250000

        grad_manual[0][0][1][1] = 0.112500
        grad_manual[0][1][1][1] = 0.147500
        grad_manual[1][0][1][1] = 0.080000
        grad_manual[1][1][1][1] = 0.087500

        grad_manual[0][0][1][2] = 0.072500
        grad_manual[0][1][1][2] = -0.127500
        grad_manual[1][0][1][2] = -0.070000
        grad_manual[1][1][1][2] = -0.085000

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'leakyrelu'
    def test_grad_w_conv_stride_1_golovko_leakyrelu(self):
        args = CustomGradientSDC1Arguments_stride_1()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 4

        grad = grad_w_conv_stride_1(x0, x1, y0, y1, w_shape, d_leakyrelu_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0][0][0][0] = 0.0925000
        grad_manual[0][1][0][0] = -0.1075000
        grad_manual[1][0][0][0] = -0.0600000
        grad_manual[1][1][0][0] = 0.0050000

        grad_manual[0][0][0][1] = 0.2445000
        grad_manual[0][1][0][1] = 0.0075000
        grad_manual[1][0][0][1] = 0.1535000
        grad_manual[1][1][0][1] = 0.1145000

        grad_manual[0][0][0][2] = 0.0900000
        grad_manual[0][1][0][2] = -0.1350000
        grad_manual[1][0][0][2] = 0.0050000
        grad_manual[1][1][0][2] = 0.0200000

        grad_manual[0][0][1][0] = 0.0605000
        grad_manual[0][1][1][0] = -0.1375000
        grad_manual[1][0][1][0] = -0.0100000
        grad_manual[1][1][1][0] = -0.1250000

        grad_manual[0][0][1][1] = 0.098500
        grad_manual[0][1][1][1] = 0.131500
        grad_manual[1][0][1][1] = 0.070000
        grad_manual[1][1][1][1] = 0.077500

        grad_manual[0][0][1][2] = 0.048500
        grad_manual[0][1][1][2] = -0.127500
        grad_manual[1][0][1][2] = -0.070000
        grad_manual[1][1][1][2] = -0.085000

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'sigmoid'
    def test_grad_w_conv_stride_1_golovko_sigmoid(self):
        args = CustomGradientSDC1Arguments_stride_1()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 3

        grad = grad_w_conv_stride_1(x0, x1, y0, y1, w_shape, d_sigmoid_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0][0][0][0] = -0.0011500
        grad_manual[0][1][0][0] = -0.0134500
        grad_manual[1][0][0][0] = -0.0126500
        grad_manual[1][1][0][0] = 0.0076000

        grad_manual[0][0][0][1] = 0.045625
        grad_manual[0][1][0][1] = 0.010775
        grad_manual[1][0][0][1] = 0.032550
        grad_manual[1][1][0][1] = 0.028150

        grad_manual[0][0][0][2] = 0.0109500
        grad_manual[0][1][0][2] = -0.0171500
        grad_manual[1][0][0][2] = -0.0018750
        grad_manual[1][1][0][2] = 0.0118750

        grad_manual[0][0][1][0] = 0.0126000
        grad_manual[0][1][1][0] = -0.0401250
        grad_manual[1][0][1][0] = -0.0030000
        grad_manual[1][1][1][0] = -0.0303250

        grad_manual[0][0][1][1] = 0.015800
        grad_manual[0][1][1][1] = 0.024425
        grad_manual[1][0][1][1] = 0.016025
        grad_manual[1][1][1][1] = 0.013825

        grad_manual[0][0][1][2] = 0.0077500
        grad_manual[0][1][1][2] = -0.0322000
        grad_manual[1][0][1][2] = -0.0111500
        grad_manual[1][1][1][2] = -0.0213750

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'tanh'
    def test_grad_w_conv_stride_1_golovko_tanh(self):
        args = CustomGradientSDC1Arguments_stride_1()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 3

        grad = grad_w_conv_stride_1(x0, x1, y0, y1, w_shape, d_tanh_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0][0][0][0] = -0.0941500
        grad_manual[0][1][0][0] = -0.1194500
        grad_manual[1][0][0][0] = -0.1241500
        grad_manual[1][1][0][0] = -0.0084000

        grad_manual[0][0][0][1] = 0.024875
        grad_manual[0][1][0][1] = -0.074475
        grad_manual[1][0][0][1] = 0.055050
        grad_manual[1][1][0][1] = 0.040650

        grad_manual[0][0][0][2] = -0.049550
        grad_manual[0][1][0][2] = -0.127650
        grad_manual[1][0][0][2] = -0.037125
        grad_manual[1][1][0][2] = 0.013125

        grad_manual[0][0][1][0] = -0.126400
        grad_manual[0][1][1][0] = -0.204375
        grad_manual[1][0][1][0] = -0.021000
        grad_manual[1][1][1][0] = -0.165075

        grad_manual[0][0][1][1] = -0.0282000
        grad_manual[0][1][1][1] = -0.0233250
        grad_manual[1][0][1][1] = -0.0092250
        grad_manual[1][1][1][1] = -0.0254250

        grad_manual[0][0][1][2] = -0.107250
        grad_manual[0][1][1][2] = -0.161700
        grad_manual[1][0][1][2] = -0.070650
        grad_manual[1][1][1][2] = -0.113625

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    """Test function 'grad_w_conv_stride_n()'
    Using Hinton and Golovko approaches. Testing 'relu', 'leakyrelu', 'sigmoid' and 'tanh' function activations"""

    # test only hinton method
    def test_grad_w_conv_stride_n_hinton(self):
        args = CustomGradientSDC1Arguments_stride_n()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        d_act = lambda var: 1.0

        accuracy = 4

        grad = grad_w_conv_stride_n(x0, x1, y0, y1, w_shape, stride=2, d_act=d_act)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0][0][0][0] = -0.200000
        grad_manual[0][1][0][0] = -0.177500
        grad_manual[1][0][0][0] = -0.185000
        grad_manual[1][1][0][0] = -0.065000

        grad_manual[0][0][0][1] = -0.092500
        grad_manual[0][1][0][1] = -0.292500
        grad_manual[1][0][0][1] = -0.257500
        grad_manual[1][1][0][1] = -0.175000

        grad_manual[0][0][0][2] = -0.137500
        grad_manual[0][1][0][2] = 0.012500
        grad_manual[1][0][0][2] = -0.372500
        grad_manual[1][1][0][2] = -0.065000

        grad_manual[0][0][1][0] = 0.087500
        grad_manual[0][1][1][0] = -0.062500
        grad_manual[1][0][1][0] = -0.052500
        grad_manual[1][1][1][0] = 0.032500

        grad_manual[0][0][1][1] = 0.0050000
        grad_manual[0][1][1][1] = -0.2525000
        grad_manual[1][0][1][1] = -0.1850000
        grad_manual[1][1][1][1] = -0.2675000

        grad_manual[0][0][1][2] = 0.092500
        grad_manual[0][1][1][2] = -0.080000
        grad_manual[1][0][1][2] = -0.130000
        grad_manual[1][1][1][2] = -0.177500

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'relu'
    def test_grad_w_conv_stride_n_golovko_relu(self):
        args = CustomGradientSDC1Arguments_stride_n()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 4

        grad = grad_w_conv_stride_n(x0, x1, y0, y1, w_shape, stride=2, d_act=d_relu_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0,0,0,0] = 0.055000
        grad_manual[0,1,0,0] = -0.112500
        grad_manual[1,0,0,0] = -0.170000
        grad_manual[1,1,0,0] = -0.020000
        grad_manual[0,0,1,0] = 0.137500
        grad_manual[0,1,1,0] = -0.012500
        grad_manual[1,0,1,0] = -0.012500
        grad_manual[1,1,1,0] = 0.077500
        grad_manual[0,0,0,1] = -0.020000
        grad_manual[0,1,0,1] = -0.212500
        grad_manual[1,0,0,1] = -0.237500
        grad_manual[1,1,0,1] = -0.155000
        grad_manual[0,0,0,2] = 0.010000
        grad_manual[0,1,0,2] = 0.012500
        grad_manual[1,0,0,2] = -0.372500
        grad_manual[1,1,0,2] = -0.065000
        grad_manual[0,0,1,1] = 0.065000
        grad_manual[0,1,1,1] = -0.162500
        grad_manual[1,0,1,1] = -0.125000
        grad_manual[1,1,1,1] = -0.187500
        grad_manual[0,0,1,2] = 0.092500
        grad_manual[0,1,1,2] = -0.080000
        grad_manual[1,0,1,2] = -0.130000
        grad_manual[1,1,1,2] = -0.177500

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'leakyrelu'
    def test_grad_w_conv_stride_n_golovko_leakyrelu(self):
        args = CustomGradientSDC1Arguments_stride_n()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 4

        grad = grad_w_conv_stride_n(x0, x1, y0, y1, w_shape, stride=2, d_act=d_leakyrelu_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0,0,0,0] = 0.004000
        grad_manual[0,1,0,0] = -0.125500
        grad_manual[1,0,0,0] = -0.173000
        grad_manual[1,1,0,0] = -0.029000
        grad_manual[0,0,1,0] = 0.127500
        grad_manual[0,1,1,0] = -0.022500
        grad_manual[1,0,1,0] = -0.020500
        grad_manual[1,1,1,0] = 0.068500
        grad_manual[0,0,0,1] = -0.034500
        grad_manual[0,1,0,1] = -0.228500
        grad_manual[1,0,0,1] = -0.241500
        grad_manual[1,1,0,1] = -0.159000
        grad_manual[0,0,0,2] = -0.019500
        grad_manual[0,1,0,2] = 0.012500
        grad_manual[1,0,0,2] = -0.372500
        grad_manual[1,1,0,2] = -0.065000
        grad_manual[0,0,1,1] = 0.053000
        grad_manual[0,1,1,1] = -0.180500
        grad_manual[1,0,1,1] = -0.137000
        grad_manual[1,1,1,1] = -0.203500
        grad_manual[0,0,1,2] = 0.092500
        grad_manual[0,1,1,2] = -0.080000
        grad_manual[1,0,1,2] = -0.130000
        grad_manual[1,1,1,2] = -0.177500

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'sigmoid'
    def test_grad_w_conv_stride_n_golovko_sigmoid(self):
        args = CustomGradientSDC1Arguments_stride_n()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 3

        grad = grad_w_conv_stride_n(x0, x1, y0, y1, w_shape, stride=2, d_act=d_sigmoid_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0,0,0,0] = 0.010150
        grad_manual[0,1,0,0] = -0.002250
        grad_manual[1,0,0,0] = -0.008925
        grad_manual[1,1,0,0] = -0.002700
        grad_manual[0,0,1,0] = 0.025050
        grad_manual[0,1,1,0] = 0.000400
        grad_manual[1,0,1,0] = -0.002200
        grad_manual[1,1,1,0] = 0.020250
        grad_manual[0,0,0,1] = -0.009500
        grad_manual[0,1,0,1] = -0.019075
        grad_manual[1,0,0,1] = -0.028475
        grad_manual[1,1,0,1] = -0.036800
        grad_manual[0,0,0,2] = 0.003325
        grad_manual[0,1,0,2] = 0.007975
        grad_manual[1,0,0,2] = -0.041600
        grad_manual[1,1,0,2] = -0.007350
        grad_manual[0,0,1,1] = 0.009175
        grad_manual[0,1,1,1] = -0.034675
        grad_manual[1,0,1,1] = -0.027675
        grad_manual[1,1,1,1] = -0.028725
        grad_manual[0,0,1,2] = 0.018175
        grad_manual[0,1,1,2] = -0.013525
        grad_manual[1,0,1,2] = -0.025050
        grad_manual[1,1,1,2] = -0.018175

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'tanh'
    def test_grad_w_conv_stride_n_golovko_tanh(self):
        args = CustomGradientSDC1Arguments_stride_n()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        accuracy = 3

        grad = grad_w_conv_stride_n(x0, x1, y0, y1, w_shape, stride=2, d_act=d_tanh_cuda)
        grad_round = round_4d(grad, accuracy=accuracy)

        grad_manual = np.zeros(grad.shape)

        grad_manual[0,0,0,0] = -0.229350
        grad_manual[0,1,0,0] = -0.190250
        grad_manual[1,0,0,0] = -0.201675
        grad_manual[1,1,0,0] = -0.068700
        grad_manual[0,0,1,0] = 0.012050
        grad_manual[0,1,1,0] = -0.050100
        grad_manual[1,0,1,0] = -0.059700
        grad_manual[1,1,1,0] = -0.006750
        grad_manual[0,0,0,1] = -0.107000
        grad_manual[0,1,0,1] = -0.317825
        grad_manual[1,0,0,1] = -0.260225
        grad_manual[1,1,0,1] = -0.177800
        grad_manual[0,0,0,2] = -0.153425
        grad_manual[0,1,0,2] = -0.034775
        grad_manual[1,0,0,2] = -0.371100
        grad_manual[1,1,0,2] = -0.092850
        grad_manual[0,0,1,1] = -0.041575
        grad_manual[0,1,1,1] = -0.213425
        grad_manual[1,0,1,1] = -0.179425
        grad_manual[1,1,1,1] = -0.281975
        grad_manual[0,0,1,2] = 0.029925
        grad_manual[0,1,1,2] = -0.091775
        grad_manual[1,0,1,2] = -0.125050
        grad_manual[1,1,1,2] = -0.182925

        grad_manual = round_4d(grad_manual, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    """Test function 'grad_w_fc()'
    Using Hinton and Golovko approaches. Testing 'relu', 'leakyrelu', 'sigmoid' and 'tanh' function activations"""

    # test only hinton method
    def test_grad_w_fc_hinton(self):
        args = CustomGradientSDC1Arguments_fc()

        x0 = args._x0
        x1 = args._x1

        y0 = args._y0
        y1 = args._y1

        w_shape = [5, 3]

        d_act = lambda var: 1.0
        accuracy = 4

        grad_manual = np.zeros(w_shape)
        grad_manual[0,0] = 0.220000
        grad_manual[0,1] = 0.340000
        grad_manual[0,2] = 0.200000
        grad_manual[1,0] = -0.020000
        grad_manual[1,1] = 0.070000
        grad_manual[1,2] = 0.050000
        grad_manual[2,0] = 0.240000
        grad_manual[2,1] = 0.270000
        grad_manual[2,2] = 0.150000
        grad_manual[3,0] = -0.020000
        grad_manual[3,1] = 0.440000
        grad_manual[3,2] = 0.300000
        grad_manual[4,0] = 0.020000
        grad_manual[4,1] = 0.670000
        grad_manual[4,2] = 0.450000

        grad = grad_w_fc(x0, x1, y0, y1, w_shape, grad_name='conv1', prefix='recovery', d_act=d_act)
        grad_round = round_2d(grad, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'relu'
    def test_grad_w_fc_golovko_relu(self):
        args = CustomGradientSDC1Arguments_fc()

        x0 = args._x0
        x1 = args._x1

        y0 = args._y0
        y1 = args._y1

        w_shape = [5, 3]

        accuracy = 4

        grad_manual = np.zeros(w_shape)
        grad_manual[0,0] = 0.220000
        grad_manual[0,1] = 0.340000
        grad_manual[0,2] = 0.200000
        grad_manual[1,0] = -0.020000
        grad_manual[1,1] = 0.070000
        grad_manual[1,2] = 0.050000
        grad_manual[2,0] = 0.240000
        grad_manual[2,1] = 0.270000
        grad_manual[2,2] = 0.150000
        grad_manual[3,0] = -0.020000
        grad_manual[3,1] = 0.440000
        grad_manual[3,2] = 0.300000
        grad_manual[4,0] = 0.020000
        grad_manual[4,1] = 0.670000
        grad_manual[4,2] = 0.450000

        grad = grad_w_fc(x0, x1, y0, y1, w_shape, grad_name='conv1', prefix='recovery', d_act=d_relu_cuda)
        grad_round = round_2d(grad, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'leakyrelu'
    def test_grad_w_fc_golovko_leakyrelu(self):
        args = CustomGradientSDC1Arguments_fc()

        x0 = args._x0
        x1 = args._x1

        y0 = args._y0
        y1 = args._y1

        w_shape = [5, 3]

        accuracy = 4

        grad_manual = np.zeros(w_shape)
        grad_manual[0,0] = 0.220000
        grad_manual[0,1] = 0.340000
        grad_manual[0,2] = 0.200000
        grad_manual[1,0] = -0.020000
        grad_manual[1,1] = 0.070000
        grad_manual[1,2] = 0.050000
        grad_manual[2,0] = 0.240000
        grad_manual[2,1] = 0.270000
        grad_manual[2,2] = 0.150000
        grad_manual[3,0] = -0.020000
        grad_manual[3,1] = 0.440000
        grad_manual[3,2] = 0.300000
        grad_manual[4,0] = 0.020000
        grad_manual[4,1] = 0.670000
        grad_manual[4,2] = 0.450000

        grad = grad_w_fc(x0, x1, y0, y1, w_shape, grad_name='conv1', prefix='recovery', d_act=d_leakyrelu_cuda)
        grad_round = round_2d(grad, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'sigmoid'
    def test_grad_w_fc_golovko_sigmoid(self):
        args = CustomGradientSDC1Arguments_fc()

        x0 = args._x0
        x1 = args._x1

        y0 = args._y0
        y1 = args._y1

        w_shape = [5, 3]

        accuracy = 4

        grad_manual = np.zeros(w_shape)
        grad_manual[0,0] = 0.059200
        grad_manual[0,1] = 0.039600
        grad_manual[0,2] = 0.050000
        grad_manual[1,0] = -0.003200
        grad_manual[1,1] = 0.006300
        grad_manual[1,2] = 0.012500
        grad_manual[2,0] = 0.053400
        grad_manual[2,1] = 0.031500
        grad_manual[2,2] = 0.037500
        grad_manual[3,0] = 0.004800
        grad_manual[3,1] = 0.042600
        grad_manual[3,2] = 0.075000
        grad_manual[4,0] = -0.010800
        grad_manual[4,1] = 0.060300
        grad_manual[4,2] = 0.112500

        grad = grad_w_fc(x0, x1, y0, y1, w_shape, grad_name='conv1', prefix='recovery', d_act=d_sigmoid_cuda)
        grad_round = round_2d(grad, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    # test golovko method with activation 'tanh'
    def test_grad_w_fc_golovko_tanh(self):
        args = CustomGradientSDC1Arguments_fc()

        x0 = args._x0
        x1 = args._x1

        y0 = args._y0
        y1 = args._y1

        w_shape = [5, 3]

        accuracy = 4

        grad_manual = np.zeros(w_shape)
        grad_manual[0,0] = 0.223200
        grad_manual[0,1] = 0.103600
        grad_manual[0,2] = 0.150000
        grad_manual[1,0] = -0.007200
        grad_manual[1,1] = 0.013300
        grad_manual[1,2] = 0.037500
        grad_manual[2,0] = 0.251400
        grad_manual[2,1] = 0.094500
        grad_manual[2,2] = 0.112500
        grad_manual[3,0] = 0.020800
        grad_manual[3,1] = 0.092600
        grad_manual[3,2] = 0.225000
        grad_manual[4,0] = -0.026800
        grad_manual[4,1] = 0.127300
        grad_manual[4,2] = 0.337500

        grad = grad_w_fc(x0, x1, y0, y1, w_shape, grad_name='conv1', prefix='recovery', d_act=d_tanh_cuda)
        grad_round = round_2d(grad, accuracy=accuracy)
        self.assertTrue(np.equal(grad_round, grad_manual).all())

    """Test function 'grad_bias_x()'"""

    def test_grad_bias_x_hinton(self):
        args = CustomGradientSDC1Arguments_stride_1()

        # x shape is (NHWQ). Our x shape is (1, 2, 2, 2)
        x0 = args._x0
        x1 = args._x1

        # y shape is (NHWK). Our y shape is (1, 1, 1, 3)
        y0 = args._y0
        y1 = args._y1

        # w_shape is (HWQK). Our shape is (2, 2, 2, 3)
        w_shape = [2, 2, 2, 3]

        bias_shape = [3]
        stride = 1

        d_act = lambda var: 1.0

        grad_manual = np.zeros(3)

        grad = grad_bias_x(x0, x1, y0.shape, w_shape, bias_shape, stride, d_act)
        a = 1
