from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        return array_api.log(array_api.sum(array_api.exp(Z - array_api.max(Z, self.axes, keepdims=True)), self.axes)) + array_api.max(Z, axis=self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(
            axis=self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp = exp_z.sum(self.axes)
        if self.axes:
            grad_shape = list(z.shape)
            for axis in self.axes:
                grad_shape[axis] = 1
        else:
            grad_shape = [1 for _ in range(len(z.shape))]
        return (out_grad / sum_exp).reshape(grad_shape).broadcast_to(z.shape) * exp_z
        # END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
