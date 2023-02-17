from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ._cnode import (Access, Addition, Arctangent2, Clip, ComputationalNode,
                     Concatenation, Cosine, CrossEntropyLossFromLogits,
                     Division, Exponential, Exponentiation, GaussianEntropy,
                     GaussianLogProbability, GaussianSample, HyperbolicTangent,
                     Logarithm, MatrixMultiplication, Maximum, Mean,
                     MeanSquaredErrorLoss, Minimum, Multiplication, Negation,
                     ReLU, Reshape, Sigmoid, Sine, SoftMax, SoftPlus, Split,
                     Stack, Subtraction, Summation, Variable)
from ._typing import NDArrayFloat


class Tensor:

    """Implements the backpropgation algorithm and overloads operators."""

    _computational_node: ComputationalNode

    def __init__(self, computational_node: ComputationalNode, /) -> None:
        self._computational_node = computational_node

    @staticmethod
    def normalize_operand(
            operand: Tensor | float | int | NDArrayFloat,
            /
            ) -> ComputationalNode:
        if isinstance(operand, Tensor):
            return operand.computational_node
        else:
            return Variable(operand, requires_gradient=False)

    def __getitem__(
            self,
            key: int | Tuple[int, ...] | slice | Tuple[slice, ...],
            /
            ) -> Tensor:
        result = Access((self._computational_node,), key)
        return Tensor(result)

    def __len__(self, /) -> int:
        return len(self._computational_node.result)

    def __add__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            self._computational_node, Tensor.normalize_operand(other)
        )
        result = Addition(normalized_operands)
        return Tensor(result)

    def __sub__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            self._computational_node, Tensor.normalize_operand(other)
        )
        result = Subtraction(normalized_operands)
        return Tensor(result)

    def __mul__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            self._computational_node, Tensor.normalize_operand(other)
        )
        result = Multiplication(normalized_operands)
        return Tensor(result)

    def __matmul__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            self._computational_node, Tensor.normalize_operand(other)
        )
        result = MatrixMultiplication(normalized_operands)
        return Tensor(result)

    def __truediv__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            self._computational_node, Tensor.normalize_operand(other)
        )
        result = Division(normalized_operands)
        return Tensor(result)

    def __pow__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            self._computational_node, Tensor.normalize_operand(other)
        )
        result = Exponentiation(normalized_operands)
        return Tensor(result)

    def __radd__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            Tensor.normalize_operand(other), self._computational_node
        )
        result = Addition(normalized_operands)
        return Tensor(result)

    def __rsub__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            Tensor.normalize_operand(other), self._computational_node
        )
        result = Subtraction(normalized_operands)
        return Tensor(result)

    def __rmul__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            Tensor.normalize_operand(other), self._computational_node
        )
        result = Multiplication(normalized_operands)
        return Tensor(result)

    def __rmatmul__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            Tensor.normalize_operand(other), self._computational_node
        )
        result = MatrixMultiplication(normalized_operands)
        return Tensor(result)

    def __rtruediv__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            Tensor.normalize_operand(other), self._computational_node
        )
        result = Division(normalized_operands)
        return Tensor(result)

    def __rpow__(
            self,
            other: Tensor | float | int | NDArrayFloat,
            /
            ) -> Tensor:
        normalized_operands = (
            Tensor.normalize_operand(other), self._computational_node
        )
        result = Exponentiation(normalized_operands)
        return Tensor(result)

    def __neg__(self, /) -> Tensor:
        result = Negation((self._computational_node,))
        return Tensor(result)

    def backpropagate(self, /) -> None:
        """Compute gradients according to chain rule."""

        if self._computational_node.result.size != 1:
            raise Exception('Gradient can be only computed for scalar '
                            'functions!')

        nodes: Set[ComputationalNode] = set()
        node_stack = [self._computational_node]
        while node_stack:
            node = node_stack.pop()
            if node not in nodes:
                nodes.add(node)
                node_stack.extend(node.operands)

        dependency_counters = dict.fromkeys(nodes, 0)
        for node in nodes:
            for operand_node in node.operands:
                dependency_counters[operand_node] += 1

        backpropagation_order: List[ComputationalNode] = list()
        node_stack = [self._computational_node]
        while node_stack:
            node = node_stack.pop()
            backpropagation_order.append(node)

            for operand_node in node.operands:
                dependency_counters[operand_node] -= 1
                if dependency_counters[operand_node] == 0:
                    node_stack.append(operand_node)

        for node in backpropagation_order:
            if node is self._computational_node:
                node.set_unit_gradient()
            else:
                node.set_zero_gradient()

        for node in backpropagation_order:
            node.backpropagate()

    def reshape(self, shape: Tuple[int, ...], /) -> Tensor:
        result = Reshape((self._computational_node,), shape)
        return Tensor(result)

    def mean(
            self,
            axis: Optional[int | Tuple[int, ...]] = None,
            keep_dimensions: bool = False
            ) -> Tensor:
        result = Mean((self._computational_node,), axis, keep_dimensions)
        return Tensor(result)

    def sum(
            self,
            axis: Optional[int | Tuple[int, ...]] = None,
            keep_dimensions: bool = False
            ) -> Tensor:
        result = Summation((self._computational_node,), axis, keep_dimensions)
        return Tensor(result)

    def detach(self, requires_gradient: bool = False, /) -> Tensor:
        return Tensor(Variable(self._computational_node.result,
                               requires_gradient))

    @property
    def data(self, /) -> NDArrayFloat:
        return self._computational_node.result

    @data.setter
    def data(self, value: NDArrayFloat, /) -> None:
        self._computational_node.result = value

    @property
    def gradient(self, /) -> NDArrayFloat:
        return self._computational_node.gradient

    @property
    def requires_gradient(self, /) -> bool:
        return self._computational_node.requires_gradient

    @property
    def shape(self, /) -> Tuple[int, ...]:
        return self._computational_node.result.shape

    @property
    def computational_node(self, /) -> ComputationalNode:
        return self._computational_node


def tensor(
        value: float | int | NDArrayFloat,
        requires_gradient: bool = True
        ) -> Tensor:
    return Tensor(Variable(value, requires_gradient))


def exp(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = Exponential(normalized_operands)
    return Tensor(result)

def log(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = Logarithm(normalized_operands)
    return Tensor(result)


def relu(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = ReLU(normalized_operands)
    return Tensor(result)

def sigmoid(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = Sigmoid(normalized_operands)
    return Tensor(result)

def softmax(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = SoftMax(normalized_operands)
    return Tensor(result)

def softplus(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = SoftPlus(normalized_operands)
    return Tensor(result)

def tanh(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = HyperbolicTangent(normalized_operands)
    return Tensor(result)


def arctan2(
        y: Tensor | float | int | NDArrayFloat,
        x: Tensor | float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in (y, x))
    result = Arctangent2(normalized_operands)
    return Tensor(result)

def cos(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = Cosine(normalized_operands)
    return Tensor(result)

def sin(x: Tensor | float | int | NDArrayFloat, /) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = Sine(normalized_operands)
    return Tensor(result)


def concatenate(
        operands: Tuple[Tensor | NDArrayFloat, ...],
        axis: int = 0
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in operands)
    result = Concatenation(normalized_operands, axis)
    return Tensor(result)

def split(
        operand: Tensor | NDArrayFloat,
        split_size: int,
        axis: int
        ) -> Tuple[Tensor, ...]:
    normalized_operands = (Tensor.normalize_operand(operand),)

    results: List[Tensor] = []
    for i in range(operand.shape[axis] // split_size):
        indices = (slice(0, -1),)*axis \
            + (slice(i*split_size, (i + 1)*split_size),)
        result = Tensor(Split(normalized_operands, indices))
        results.append(result)

    return tuple(results)

def stack(
        operands: Tuple[Tensor | NDArrayFloat, ...],
        axis: int = 0
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in operands)
    result = Stack(normalized_operands, axis)
    return Tensor(result)


def clip(
        x: Tensor | float | int | NDArrayFloat,
        minimum_value: float | int | NDArrayFloat,
        maximum_value: float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(x),)
    result = Clip(normalized_operands, minimum_value, maximum_value)
    return Tensor(result)

def maximum(
        x_1: Tensor | float | int | NDArrayFloat,
        x_2: Tensor | float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in (x_1, x_2))
    result = Maximum(normalized_operands)
    return Tensor(result)

def minimum(
        x_1: Tensor | float | int | NDArrayFloat,
        x_2: Tensor | float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in (x_1, x_2))
    result = Minimum(normalized_operands)
    return Tensor(result)


def cross_entropy_loss_from_logits(
        z: Tensor | float | int | NDArrayFloat,
        true_y: Tensor | float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in (z, true_y))
    result = CrossEntropyLossFromLogits(normalized_operands)
    return Tensor(result)

def mean_squared_error_loss(
        y: Tensor | float | int | NDArrayFloat,
        true_y: Tensor | float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in (y, true_y))
    result = MeanSquaredErrorLoss(normalized_operands)
    return Tensor(result)


def gaussian_entropy(
        standard_deviation: Tensor | float | int | NDArrayFloat,
        /
        ) -> Tensor:
    normalized_operands = (Tensor.normalize_operand(standard_deviation),)
    result = GaussianEntropy(normalized_operands)
    return Tensor(result)

def gaussian_log_probability(
        sample: Tensor | float | int | NDArrayFloat,
        mean: Tensor | float | int | NDArrayFloat,
        standard_deviation: Tensor | float | int | NDArrayFloat
        ) -> Tensor:
    normalized_operands = tuple(
        Tensor.normalize_operand(operand)
        for operand in (sample, mean, standard_deviation))
    result = GaussianLogProbability(normalized_operands)
    return Tensor(result)

def gaussian_sample(
        mean: Tensor | float | int | NDArrayFloat,
        standard_deviation: Tensor | float | int | NDArrayFloat
        ) -> Tensor:
    normalized_operands = tuple(Tensor.normalize_operand(operand)
                                for operand in (mean, standard_deviation))
    result = GaussianSample(normalized_operands)
    return Tensor(result)