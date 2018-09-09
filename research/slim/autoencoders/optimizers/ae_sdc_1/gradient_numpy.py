import numpy as np
from research.slim.autoencoders.optimizers.optimizer_utils import Formulas, LayerShapeType, padding_fn, d_act_dic
from research.slim.autoencoders.optimizers.ae_sdc_1.cuda_fn import grad_weight, grad_bias


class GradientSDC1:

    def __init__(self, grads, x, y, stride, padding, formulas=Formulas, activation_name=None,
                 ae_layer_prefix='recover', input_shape_type=LayerShapeType.NHWC):
        self._grads = grads
        self._y = y
        self._x = x
        self._stride = stride
        self._padding = padding
        self._prefix = ae_layer_prefix
        self._input_shape_type = input_shape_type
        self._d_act = d_act_dic[activation_name] if formulas == Formulas.golovko else lambda y: 1.0

    def grad_weight(self, grad):
        result = np.zeros(grad.shape.as_list(), np.float32)
        grad_weight(x=self._x,
                    y=self._y,
                    w_shape=grad.shape.as_list(),
                    stride=self._stride,
                    d_act=self._d_act)
        return result

    def grad_biases(self, grads, grad_name):
        return grad_bias(x=self._x,
                         y=self._y,
                         stride=self._stride,
                         grads=grads,
                         grad_name=grad_name,
                         prefix=self._prefix,
                         d_act=self._d_act)

    # def grad_biases_x(self, grads, grad_name, xe):
    #     y = self._y
    #     x = self._x
    #     stride = self._stride
    #     prefix = self._prefix
    #     d_act = self._d_act
    #
    #     x_shape = x[0].shape
    #     y_shape = y[0].shape
    #
    #     # Get weights_shape for x1 layers
    #     w_shape = None
    #     for name in grads:
    #         if name.find(prefix) != -1 and name.find('weight') != -1:
    #             w_shape = grads[name].shape.as_list()
    #             break
    #
    #     grad_value = np.zeros(grads[grad_name].shape, np.float32)
    #
    #     # Convolution layers
    #     if len(x_shape) == 4:
    #         for q in range(x_shape[0]):
    #             t = []
    #             for c in range(x_shape[3]):
    #                 for m in range(w_shape[0]):
    #                     for n in range(w_shape[1]):
    #                         for i in range(y_shape[0]):
    #                             for j in range(y_shape[1]):
    #                                 t.append(xe[q][i * stride + m][j * stride + n][c] * \
    #                                          d_act(x[1][q][i * stride + m][j * stride + n][c]))
    #             grad_value[q] = np.sum(t)
    #
    #     # Full connections layers: k = q = batch_size
    #     else:
    #         for k in range(x_shape[0]):
    #             for i in range(x_shape[1]):
    #                 grad_value[i] = xe[k][i] * d_act(x[1][k][i])
    #
    #     return grad_value

    # def grad_biases_y(self, grad, ye):
    #     y = self._y
    #     d_act = self._d_act
    #
    #     y_shape = y[0].shape
    #
    #     grad_value = np.zeros(grad.shape, dtype=np.float32)
    #
    #     # Convolution layers
    #     if len(y_shape) == 4:
    #         for k in range(y_shape[3]):
    #             t = []
    #             for c in range(y_shape[0]):
    #                 for i in range(y_shape[1]):
    #                     for j in range(y_shape[2]):
    #                         t.append(ye[c][i][j][k] * \
    #                                  d_act(y[1][c][i][j][k]))
    #             grad_value[k] = np.sum(t)
    #
    #     # Full connections layers: k = q = batch_size
    #     else:
    #         for k in range(y_shape[0]):
    #             for j in range(y_shape[1]):
    #                 grad_value[j] = ye[k][j] * d_act(y[1][k][j])
    #
    #     return grad_value

    def run(self):
        grads = self._grads
        y = self._y
        x = self._x
        padding = self._padding
        stride = self._stride
        prefix = self._prefix
        input_shape_type = self._input_shape_type

        # If input shape type is CHWN transpose it to NHWC
        if input_shape_type == LayerShapeType.CHWN:
            x[0] = np.transpose(x[0], axes=[3, 1, 2, 0])
            x[1] = np.transpose(x[1], axes=[3, 1, 2, 0])

        # Check layers shape
        assert x[0].shape == x[1].shape
        assert y[0].shape == y[1].shape
        assert len(y[0].shape) == len(x[0].shape) == 4 or len(y[0].shape) == len(x[0].shape) == 2

        if padding == 'SAME':

            # layer_type = 'x' or 'y'
            # Find x_weights for layer_type = x
            # Find y_weights for layer_type = y
            def var_by_layer(layer_type, grads, prefix):
                for grad_name in grads:
                    if grad_name.find('weight') != -1:
                        if layer_type == 'x' and grad_name.find(prefix) != -1:
                            return grads[grad_name]
                        else:
                            return grads[grad_name]

            """ Change x0, x1 layers: append padding to maps as zeros sides """
            x_weight_shape = var_by_layer('x', grads, prefix)._shape.as_list()
            x[0] = padding_fn(layer=x[0], w_shape=x_weight_shape, stride=stride, layer_type='x')
            x[1] = padding_fn(layer=x[1], w_shape=x_weight_shape, stride=stride, layer_type='x')

        ye = y[1] - y[0]
        xe = x[1] - x[0]

        grads_value = {}

        for grad_name in grads:

            # Gradient for weights
            if grad_name.find('weight') != -1:
                grad = grads[grad_name]
                grad_value = self.grad_weight(grad)

            # Gradient for biases
            else:
                grad_value = self.grad_biases(grads, grad_name)
                # Gradient for input_sdx_1 layer
                # if grad_name.find(prefix) != -1:
                #     grad_value = self.grad_biases_x(grads, grad_name, xe)
                # # Gradient for output_sdc_0 layer
                # else:
                #     grad_value = self.grad_biases_y(grads[grad_name], ye)

            grads_value[grads[grad_name]] = grad_value

        return grads_value
