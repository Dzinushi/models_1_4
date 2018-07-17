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


def derivate_activation(activation_name):
    try:
        return activation_dic[activation_name]
    except KeyError as e:
        # можно также присвоить значение по умолчанию вместо бросания исключения
        raise ValueError('Undefined unit: {}'.format(e.args[0]))


# With derivative
class GradientDescentOptimizer_2(optimizer.Optimizer):
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
        super(GradientDescentOptimizer_2, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._activation = derivate_activation(activation_name)

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._activation_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr,
                                           name="learning_rate")
        self._activation_t = ops.convert_to_tensor(self._activation,
                                                   name='activation')

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        activation_t = math_ops.cast(self._activation_t, var.dtype.base_dtype)

        var_update = state_ops.assign_sub(var, lr_t * grad * activation_t)
        return var_update

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
