from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

activation_dic = {
    'relu': 1.0,
    'leaky_relu': 0.2
}


def derivative_activation(activation_name):
    try:
        return activation_dic[activation_name]
    except KeyError as e:
        # можно также присвоить значение по умолчанию вместо бросания исключения
        raise ValueError('Undefined unit: {}'.format(e.args[0]))


def reshape_2d(tensor, axis=0):
    # if len(tensor.shape) != 4:
    #     return tensor
    shape = tensor.shape
    mult = 1
    for dimension in shape:
        if dimension.value != 1:
            mult *= dimension.value
    if axis == 0:
        tennsor_2d = tf.reshape(tensor, shape=(1, mult))
    elif axis == 1:
        tennsor_2d = tf.reshape(tensor, shape=(mult, 1))
    else:
        raise ValueError('Axis must have 2 dimensional (0 or 1): {}'.format(axis))
    return tennsor_2d


# # With derivative
# class GradientDescentOptimizerSDC1(optimizer.Optimizer):
#     """Optimizer that implements the gradient descent algorithm.
#     """
#
#     def __init__(self, learning_rate, activation_name, loss_map, use_locking=False, name="GradientDescent"):
#         """Construct a new gradient descent optimizer.
#
#         Args:
#           learning_rate: A Tensor or a floating point value.  The learning
#             rate to use.
#           use_locking: If True use locks for update operations.
#           name: Optional name prefix for the operations created when applying
#             gradients. Defaults to "GradientDescent".
#         """
#         super(GradientDescentOptimizerSDC1, self).__init__(use_locking, name)
#         self._lr = learning_rate
#         self._activation = derivative_activation(activation_name)
#         # shape = (1 x N)
#         self._x_0 = reshape_2d(loss_map[0]['input'], axis=0)
#         self._x_1 = reshape_2d(loss_map[0]['output'], axis=0)
#         # shape = (N x 1)
#         self._y_0 = reshape_2d(loss_map[1]['input'], axis=1)
#         self._y_1 = reshape_2d(loss_map[1]['output'], axis=1)
#
#         # Tensor versions of the constructor arguments, created in _prepare().
#         self._lr_t = None
#         self._activation_t = None
#         self._x_0_t = None
#         self._x_1_t = None
#         self._y_0_t = None
#         self._y_1_t = None
#
#     def _prepare(self):
#         self._lr_t = ops.convert_to_tensor(self._lr,
#                                            name="learning_rate")
#         self._activation_t = ops.convert_to_tensor(self._activation,
#                                                    name='derivative_activation')
#         self._x_0_t = ops.convert_to_tensor(self._x_0,
#                                             name='input_sdc_0')
#         self._x_1_t = ops.convert_to_tensor(self._x_1,
#                                             name='input_sdc_1')
#         self._y_0_t = ops.convert_to_tensor(self._y_0,
#                                             name='hidden_sdc_0')
#         self._y_1_t = ops.convert_to_tensor(self._y_1,
#                                             name='hidden_sdc_1')
#
#     def _apply_dense(self, grad, var):
#         lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
#         activation_t = math_ops.cast(self._activation_t, var.dtype.base_dtype)
#         x_0_t = math_ops.cast(self._x_0_t, var.dtype.base_dtype)
#         x_1_t = math_ops.cast(self._x_1_t, var.dtype.base_dtype)
#         y_0_t = math_ops.cast(self._y_0_t, var.dtype.base_dtype)
#         y_1_t = math_ops.cast(self._y_1_t, var.dtype.base_dtype)
#
#         var_update = state_ops.assign_sub(var,
#                                           tf.reduce_sum(lr_t * (tf.matmul((y_1_t - y_0_t), x_1_t) * activation_t +
#                                                                 tf.matmul(y_0_t, (x_1_t - x_0_t)) * activation_t)))
#         return var_update
#
#     def _apply_sparse(self, grad, var):
#         raise NotImplementedError("Sparse gradient updates are not supported.")


# With derivative
class GradientDescentOptimizerSDC1(optimizer.Optimizer):
    """Optimizer that implements the gradient descent algorithm.
    """

    def __init__(self, learning_rate, activation_name, use_locking=False, name="GradientDescent"):
        """Construct a new gradient descent optimizer.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning
            rate to use.
          use_locking: If True use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "GradientDescent".
        """
        super(GradientDescentOptimizerSDC1, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._activation = derivative_activation(activation_name)

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._activation_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr,
                                           name="learning_rate")
        self._activation_t = ops.convert_to_tensor(self._activation,
                                                   name='derivative_activation')

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        activation_t = math_ops.cast(self._activation_t, var.dtype.base_dtype)

        var_update = state_ops.assign_sub(var, lr_t * grad * activation_t)
        return var_update

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
