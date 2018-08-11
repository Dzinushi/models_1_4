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
import numpy as np

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


# With derivative
class GradientDescentOptimizerSDC1(optimizer.Optimizer):
    """Optimizer that implements the gradient descent algorithm.
    """

    def __init__(self, grad, learning_rate, use_locking=False, name="GradientDescent"):
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
        self._grad = grad

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr,
                                           name="learning_rate")

    def _apply_dense(self, grad, var):
        lr = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        grad_custom = self._grad

        return state_ops.assign_sub(var, lr * grad_custom[var.name])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
