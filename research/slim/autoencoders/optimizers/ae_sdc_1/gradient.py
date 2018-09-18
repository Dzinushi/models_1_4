from autoencoders.optimizers.optimizer_utils import Formulas, LayerShapeType, padding_fn
from autoencoders.optimizers.ae_sdc_1.gradient_utils import grad_w_conv_stride_1, grad_w_conv_stride_n, grad_w_fc, grad_bias_x, grad_bias_y
from autoencoders.optimizers.activations import d_act_dic


class CustomGradientSDC1:

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
        y_shape = self._y[0].shape
        w_shape = grad.shape.as_list()
        d_act = self._d_act
        stride = self._stride
        x = self._x
        y = self._y
        prefix = self._prefix

        if len(y_shape) == 4:
            if stride == 1:
                return grad_w_conv_stride_1(x[0],
                                            x[1],
                                            y[0],
                                            y[1],
                                            w_shape,
                                            d_act)
            else:
                return grad_w_conv_stride_n(x[0],
                                            x[1],
                                            y[0],
                                            y[1],
                                            w_shape,
                                            stride,
                                            d_act)
        elif len(y_shape) == 2:
            return grad_w_fc(x[0],
                             x[1],
                             y[0],
                             y[1],
                             w_shape,
                             grad.name,
                             prefix,
                             d_act)
        else:
            raise ValueError('y_shape is {}. Must be 2d or 4d'.format(len(y_shape)))

    def grad_biases(self, grads, grad_name):

        prefix = self._prefix
        d_act = self._d_act
        stride = self._stride
        x = self._x
        y = self._y

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

    def run(self):
        grads = self._grads
        y = self._y
        x = self._x
        padding = self._padding
        stride = self._stride

        # Check layers shape
        assert x[0].shape == x[1].shape
        assert y[0].shape == y[1].shape
        assert len(y[0].shape) == len(x[0].shape) == 4 or len(y[0].shape) == len(x[0].shape) == 2

        if padding == 'SAME':

            # Find x_weights for layer_type = x
            # Find y_weights for layer_type = y
            def var_by_layer(grads):
                for grad_name in grads:
                    if grad_name.find('weight') != -1:
                        return grads[grad_name]

            """ Change x0, x1 layers: append padding to maps as zeros sides """
            x_weight_shape = var_by_layer(grads)._shape.as_list()
            x[0] = padding_fn(layer=x[0], w_shape=x_weight_shape, stride=stride)
            x[1] = padding_fn(layer=x[1], w_shape=x_weight_shape, stride=stride)

        grads_value = {}

        for grad_name in grads:

            # Gradient for weights
            if grad_name.find('weight') != -1:
                grad = grads[grad_name]
                grad_value = self.grad_weight(grad)

            # Gradient for biases
            else:
                grad_value = self.grad_biases(grads, grad_name)

            grads_value[grads[grad_name]] = grad_value

        return grads_value
