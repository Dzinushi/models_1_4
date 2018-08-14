from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


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
