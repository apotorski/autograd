from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import List, Optional, Tuple

import numpy as np

from ._typing import NDArrayFloat


class ComputationalNode(ABC):

    _operands: Tuple[ComputationalNode, ...]
    _result: NDArrayFloat
    _gradient: NDArrayFloat
    _requires_gradient: bool

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ...],
            result: NDArrayFloat,
            requires_gradient: bool
            ) -> None:
        self._operands = operands
        self._result = result
        self._requires_gradient = requires_gradient

        self.set_zero_gradient()

    @abstractmethod
    def backpropagate(self, /) -> None:
        ...

    def set_unit_gradient(self, /) -> None:
        self._gradient = np.ones_like(self._result, dtype=np.float32)

    def set_zero_gradient(self, /) -> None:
        self._gradient = np.zeros_like(self._result, dtype=np.float32)

    @property
    def operands(self) -> Tuple[ComputationalNode, ...]:
        return self._operands

    @property
    def result(self) -> NDArrayFloat:
        return self._result

    @result.setter
    def result(self, value: NDArrayFloat) -> None:
        self._result = value

    @property
    def requires_gradient(self) -> bool:
        return self._requires_gradient

    @requires_gradient.setter
    def requires_gradient(self, value: bool) -> None:
        self._requires_gradient = value

    @property
    def gradient(self) -> NDArrayFloat:
        return self._gradient

    @gradient.setter
    def gradient(self, value: NDArrayFloat) -> None:
        self._gradient = value


class Variable(ComputationalNode):

    def __init__(
            self,
            value: float | int | NDArrayFloat,
            requires_gradient: bool
            ) -> None:
        super().__init__(
            operands=(),
            result=value.astype(np.float32)
                   if isinstance(value, np.ndarray) else
                   np.array(value, dtype=np.float32),
            requires_gradient=requires_gradient)

    def backpropagate(self, /) -> None:
        pass


class Operation(ComputationalNode):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ...],
            result: NDArrayFloat
            ) -> None:
        super().__init__(
            operands=operands,
            result=result,
            requires_gradient=any(operand.requires_gradient
                                  for operand in operands))

    def _invert_operand_broadcast(
            self, operand_indice: int,
            operand_gradient: NDArrayFloat,
            /
            ) -> NDArrayFloat:
        operand_shape = self._operands[operand_indice].result.shape
        shape_difference = len(operand_gradient.shape) - len(operand_shape)
        padded_operand_shape = (1,)*shape_difference + operand_shape

        broadcasted_axes = tuple(
            axis for axis, (original_axis_size, broadcasted_axis_size)
            in enumerate(zip(padded_operand_shape, operand_gradient.shape))
            if original_axis_size != broadcasted_axis_size)

        summed_operand_gradient = np.sum(operand_gradient, broadcasted_axes) \
            .reshape(operand_shape)

        return summed_operand_gradient


class Access(Operation):

    _indices: int | Tuple[int, ...] | slice | Tuple[slice, ...]

    def __init__(
            self,
            operands: Tuple[ComputationalNode],
            indices: int | Tuple[int, ...] | slice | Tuple[slice, ...],
            /
            ) -> None:
        result = operands[0].result[indices]
        super().__init__(operands, result)

        self._indices = indices

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient[self._indices] += self._gradient


class Reshape(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode],
            shape: Tuple[int, ...],
            /
            ) -> None:
        result = operands[0].result.reshape(shape)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._gradient.reshape(self._operands[0].gradient.shape)


class Mean(Operation):

    _axis: Optional[int | Tuple[int, ...]]
    _keep_dimensions: bool

    def __init__(
            self,
            operands: Tuple[ComputationalNode],
            axis: Optional[int | Tuple[int, ...]] = None,
            keep_dimensions: bool = False
            ) -> None:
        result = np.mean(operands[0].result, axis=axis,
                         keepdims=keep_dimensions)
        super().__init__(operands, result)

        self._axis = axis
        self._keep_dimensions = keep_dimensions

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            if self._axis is None:
                axis_size = self._operands[0].result.size
            else:
                axis_size = self._get_axis_size(self._axis)

            if self._axis is not None and not self._keep_dimensions:
                unsqeezed_gradient = np.expand_dims(self._gradient, self._axis)
            else:
                unsqeezed_gradient = self._gradient

            self._operands[0].gradient += \
                np.ones_like(self._operands[0].gradient) / axis_size \
                    * unsqeezed_gradient

    @singledispatchmethod
    def _get_axis_size(self, axis: int | Tuple[int, ...], /) -> int:
        raise NotImplementedError

    @_get_axis_size.register(int)
    def _(self, axis: int, /) -> int:
        return self._operands[0].result.shape[axis]

    @_get_axis_size.register(tuple)
    def _(self, axes: Tuple[int, ...], /) -> int:
        return np.prod(self._operands[0].result.shape[axis] for axis in axes)


class Summation(Operation):

    _axis: Optional[int | Tuple[int, ...]]
    _keep_dimensions: bool

    def __init__(
            self,
            operands: Tuple[ComputationalNode],
            axis: Optional[int | Tuple[int, ...]] = None,
            keep_dimensions: bool = False
            ) -> None:
        result = np.sum(operands[0].result, axis=axis,
                        keepdims=keep_dimensions)
        super().__init__(operands, result)

        self._axis = axis
        self._keep_dimensions = keep_dimensions

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            if self._axis is not None and not self._keep_dimensions:
                unsqeezed_gradient = np.expand_dims(self._gradient, self._axis)
            else:
                unsqeezed_gradient = self._gradient

            self._operands[0].gradient += \
                np.ones_like(self._operands[0].gradient) \
                    * unsqeezed_gradient


class Addition(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = operands[0].result + operands[1].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        for i, operand in enumerate(self._operands):
            if operand.requires_gradient:
                operand.gradient += \
                    self._invert_operand_broadcast(i, self._gradient)


class Subtraction(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = operands[0].result - operands[1].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._invert_operand_broadcast(0, self._gradient)

        if self._operands[1].requires_gradient:
            self._operands[1].gradient -= \
                self._invert_operand_broadcast(1, self._gradient)


class Multiplication(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = operands[0].result * operands[1].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        for i, operand in enumerate(self._operands):
            if operand.requires_gradient:
                operand_gradient = \
                    self._operands[1 - i].result * self._gradient
                self._operands[i].gradient += \
                    self._invert_operand_broadcast(i, operand_gradient)


class MatrixMultiplication(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = operands[0].result @ operands[1].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            transposed_operand_1 = np.swapaxes(
                self._operands[1].result, axis1=-1, axis2=-2)
            operand_gradient = self._gradient @ transposed_operand_1
            self._operands[0].gradient += \
                self._invert_operand_broadcast(0, operand_gradient)

        if self._operands[1].requires_gradient:
            transposed_operand_0 = np.swapaxes(
                self._operands[0].result, axis1=-1, axis2=-2)
            operand_gradient = transposed_operand_0 @ self._gradient
            self._operands[1].gradient += \
                self._invert_operand_broadcast(1, operand_gradient)


class Division(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = operands[0].result / operands[1].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            operand_gradient = self._gradient / self._operands[1].result
            self._operands[0].gradient += \
                self._invert_operand_broadcast(0, operand_gradient)

        if self._operands[1].requires_gradient:
            operand_gradient = \
                self._operands[0].result / self._operands[1].result**2.0 \
                    * self._gradient
            self._operands[1].gradient += \
                self._invert_operand_broadcast(1, operand_gradient)


class Exponentiation(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = operands[0].result**operands[1].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._operands[1].result*self._operands[0].result \
                    **(self._operands[1].result - 1.0) \
                        * self._gradient

        if self._operands[1].requires_gradient:
            raise NotImplementedError


class Negation(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode],
            /
            ) -> None:
        result = -operands[0].result
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient -= self._gradient


class Exponential(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = np.exp(operands[0].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += self._result * self._gradient


class Logarithm(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = np.log(operands[0].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._gradient / self._operands[0].result


class ReLU(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = np.maximum(operands[0].result, 0.0)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                (self._operands[0].result >= 0.0) * self._gradient


class Sigmoid(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = 1.0 / (1.0 + np.exp(-operands[0].result))
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._result*(1.0 - self._result) * self._gradient


class SoftMax(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        maximum_logit = np.max(operands[0].result, axis=-2, keepdims=True)
        exponentials = np.exp(operands[0].result - maximum_logit)
        result = exponentials / np.sum(exponentials, axis=-2, keepdims=True)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            transposed_result = np.swapaxes(self._result, axis1=-1, axis2=-2)
            jacobian = \
                np.identity(self._result.shape[-2])*self._result \
                    - (self._result @ transposed_result)
            transposed_jacobian = np.swapaxes(jacobian, axis1=-1, axis2=-2)
            self._operands[0].gradient += transposed_jacobian @ self._gradient


class SoftPlus(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = \
            np.maximum(operands[0].result, 0.0) \
                + np.log1p(np.exp(-np.abs(operands[0].result)))
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._gradient / (1.0 + np.exp(-self._operands[0].result))


class HyperbolicTangent(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = np.tanh(operands[0].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                (1.0 - self._result**2.0) * self._gradient


class Arctangent2(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = np.arctan2(operands[0].result, operands[1].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        for i, operand in enumerate(self._operands):
            if operand.requires_gradient:
                operand_gradient = \
                    self._operands[1 - i].result \
                        / (self._operands[0].result**2.0 \
                            + self._operands[1].result**2.0) \
                                * self._gradient
                self._operands[i].gradient += \
                    self._invert_operand_broadcast(i, operand_gradient)


class Cosine(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = np.cos(operands[0].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient -= \
                np.sin(self._operands[0].result) \
                    * self._gradient


class Sine(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        result = np.sin(operands[0].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                np.cos(self._operands[0].result) \
                    * self._gradient


class Concatenation(Operation):

    _axis: int

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ...],
            axis: int,
            /
            ) -> None:
        result = np.concatenate([operand.result for operand in operands],
                                axis=axis)
        super().__init__(operands, result)

        self._axis = axis

    def backpropagate(self, /) -> None:
        indices: List[int] = []

        current_indice = 0
        for operand in self._operands[:-1]:
            current_indice += operand.result.shape[self._axis]
            indices.append(current_indice)

        operand_gradients = np.split(self._gradient, indices, self._axis)
        for operand, operand_gradient \
            in zip(self._operands, operand_gradients):
                if operand.requires_gradient:
                    operand.gradient += operand_gradient


class Split(Operation):

    _indices: int | Tuple[int, ...] | slice | Tuple[slice, ...]

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ...],
            indices: int | Tuple[int, ...] | slice | Tuple[slice, ...],
            /
            ) -> None:
        result = operands[0].result[indices]
        super().__init__(operands, result)

        self._indices = indices

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient[self._indices] += self._gradient


class Stack(Operation):

    _axis: int

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ...],
            axis: int,
            /
            ) -> None:
        result = np.stack([operand.result for operand in operands], axis=axis)
        super().__init__(operands, result)

        self._axis = axis

    def backpropagate(self, /) -> None:
        operand_gradients = np.split(
            self._gradient, len(self._operands), self._axis)
        for operand, operand_gradient \
            in zip(self._operands, operand_gradients):
                if operand.requires_gradient:
                    operand.gradient += operand_gradient[0]


class Clip(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode],
            minimum_value: float | int | NDArrayFloat,
            maximum_value: float | int | NDArrayFloat,
            /
            ) -> None:
        result = np.clip(operands[0].result, minimum_value, maximum_value)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                (self._operands[0].result == self._result) \
                    * self._gradient


class Maximum(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = np.maximum(operands[0].result, operands[1].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        for i, operand in enumerate(self._operands):
            if operand.requires_gradient:
                operand_gradient = \
                    (self._operands[i].result
                        >= self._operands[1 - i].result) \
                            * self._gradient
                self._operands[i].gradient += \
                    self._invert_operand_broadcast(i, operand_gradient)


class Minimum(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        result = np.minimum(operands[0].result, operands[1].result)
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        for i, operand in enumerate(self._operands):
            if operand.requires_gradient:
                operand_gradient = \
                    (self._operands[i].result
                        <= self._operands[1 - i].result) \
                            * self._gradient
                self._operands[i].gradient += \
                    self._invert_operand_broadcast(i, operand_gradient)


class CrossEntropyLossFromLogits(Operation):

    _true_y: NDArrayFloat
    _probabilites: NDArrayFloat

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        z = operands[0].result
        true_y = operands[1].result

        exponentials = np.exp(z - np.max(z, axis=-2, keepdims=True))
        probabilites = exponentials / np.sum(exponentials, axis=-2,
                                             keepdims=True)

        transposed_true_y = np.swapaxes(true_y, axis1=-1, axis2=-2)
        cross_entropy_loss = -(transposed_true_y @ np.log(probabilites))

        result = np.mean(cross_entropy_loss)
        super().__init__(operands, result)

        self._true_y = true_y
        self._probabilites = probabilites

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                (self._probabilites - self._true_y) \
                    / self._probabilites.shape[-3] \
                        * self._gradient


class MeanSquaredErrorLoss(Operation):

    _error: NDArrayFloat

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        y = operands[0].result
        true_y = operands[1].result

        error = y - true_y
        result = np.mean(error**2.0) / 2.0
        super().__init__(operands, result)

        self._error = error

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._error / self._error.size \
                    * self._gradient


class GaussianEntropy(Operation):

    def __init__(self, operands: Tuple[ComputationalNode], /) -> None:
        standard_deviation = operands[0].result

        result = 0.5*np.log(2.0*np.pi*standard_deviation**2.0) + 0.5
        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._gradient / self._operands[0].result


class GaussianLogProbability(Operation):

    def __init__(
            self,
            operands: Tuple[ComputationalNode,
                            ComputationalNode,
                            ComputationalNode],
            /
            ) -> None:
        sample = operands[0].result
        mean = operands[1].result
        standard_deviation = operands[2].result

        k = sample.shape[-2]

        result = \
            -0.5*(np.sum(((sample - mean) / standard_deviation)**2.0 \
                + 2.0*np.log(standard_deviation), axis=-2, keepdims=True) \
                    + k*np.log(2.0*np.pi))

        super().__init__(operands, result)

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            operand_gradient = \
                (self._operands[1].result - self._operands[0].result) \
                    / self._operands[2].result**2.0 \
                        * self._gradient
            self._operands[0].gradient += \
                self._invert_operand_broadcast(0, operand_gradient)

        if self._operands[1].requires_gradient:
            operand_gradient = \
                (self._operands[0].result - self._operands[1].result) \
                    / self._operands[2].result**2.0 \
                        * self._gradient
            self._operands[1].gradient += \
                self._invert_operand_broadcast(1, operand_gradient)

        if self._operands[2].requires_gradient:
            operand_gradient = \
                ((self._operands[0].result - self._operands[1].result)**2.0 \
                    - self._operands[2].result**2.0) \
                        / self._operands[2].result**3.0 \
                            * self._gradient
            self._operands[2].gradient += \
                self._invert_operand_broadcast(2, operand_gradient)


class GaussianSample(Operation):

    _epsilon: NDArrayFloat

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ComputationalNode],
            /
            ) -> None:
        mean = operands[0].result
        standard_deviation = operands[1].result

        epsilon = np.random.randn(*standard_deviation.shape)
        result = mean + standard_deviation*epsilon
        super().__init__(operands, result)

        self._epsilon = epsilon

    def backpropagate(self, /) -> None:
        if self._operands[0].requires_gradient:
            self._operands[0].gradient += \
                self._invert_operand_broadcast(0, self._gradient)

        if self._operands[1].requires_gradient:
            operand_gradient = self._epsilon * self._gradient
            self._operands[1].gradient += \
                self._invert_operand_broadcast(1, operand_gradient)