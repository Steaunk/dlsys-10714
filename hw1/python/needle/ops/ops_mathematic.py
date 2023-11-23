"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a**self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * self.scalar * (node.inputs[0]**(self.scalar-1))
        # raise NotImplementedError()
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a / b
        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, - out_grad * lhs * (rhs**-2)
        # raise NotImplementedError()
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a / self.scalar
        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        if self.axes:
            return array_api.swapaxes(a, *self.axes)
        else:
            return array_api.swapaxes(a, a.ndim-1, a.ndim-2)

        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        print("init", shape)

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        print(self.shape, a.shape, a, type(a))
        return array_api.reshape(a, self.shape)
        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        shrink_dims = [i for i in range(len(out_grad.shape))]
        for i, (a, b) in enumerate(zip(reversed(out_grad.shape), reversed(node.inputs[0].shape))):
            if a == b:
                shrink_dims[len(out_grad.shape)-i-1] = None
        shrink_dims = tuple([i for i in shrink_dims if i is not None])
        return out_grad.sum(tuple(shrink_dims)).reshape(node.inputs[0].shape)
        # raise NotImplementedError()
        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        if self.axes:
            grad_shape = list(node.inputs[0].shape)
            for axis in self.axes:
                grad_shape[axis] = 1
        else:
            grad_shape = [1 for _ in range(len(node.inputs[0].shape))]
        print(out_grad.shape, node.inputs[0].shape, grad_shape)
        return out_grad.reshape(grad_shape).broadcast_to(node.inputs[0].shape)
        # raise NotImplementedError()
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a @ b
        # return array_api.matmul(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = out_grad @ transpose(
            rhs), transpose(lhs) @ out_grad

        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(axes=tuple(
                range(len(lgrad.shape) - len(lhs.shape))))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(axes=tuple(
                range(len(rgrad.shape) - len(rhs.shape))))
        return lgrad, rgrad
        # raise NotImplementedError()
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return -a
        # raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return -out_grad
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        # raise NotImplementedError()
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        # Implement ReLU
        return array_api.maximum(a, 0)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        input = node.inputs[0].realize_cached_data()
        return out_grad * (input > 0)
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
