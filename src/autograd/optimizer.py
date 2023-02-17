from typing import List, Tuple

import numpy as np

from ._cnode import ComputationalNode
from ._typing import NDArrayFloat
from .tensor_api import Tensor


class SGD:

    _parameters: Tuple[ComputationalNode, ...]
    _learning_rate: float
    _weight_decay: float
    _momentum: float
    _dampening: float
    _nesterov_momentum: bool
    _b: List[None | NDArrayFloat]

    def __init__(
            self,
            parameters: Tuple[Tensor, ...],
            learning_rate: float = 1e-2,
            weight_decay: float = 0.0,
            momentum: float = 0.0,
            dampening: float = 0.0,
            nesterov_momentum: bool = False
            ) -> None:
        self._parameters = tuple(parameter.computational_node
                                 for parameter in parameters)

        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._momentum = momentum
        self._dampening = dampening
        self._nesterov_momentum = nesterov_momentum

        self._b = [None] * len(parameters)

    def step(self, /) -> None:
        for i, parameter in enumerate(self._parameters):
            g = parameter.gradient.copy()

            if self._weight_decay != 0.0:
                g += self._weight_decay*parameter.result

            if self._momentum != 0.0:
                if self._b[i] is None:
                    self._b[i] = g
                else:
                    self._b[i] = self._momentum*self._b[i] \
                        + (1.0 - self._dampening)*g

                if self._nesterov_momentum:
                    g += self._momentum*self._b[i]
                else:
                    g = self._b[i]

            parameter.result -= self._learning_rate*g


class RMSProp:

    _parameters: Tuple[ComputationalNode, ...]
    _learning_rate: float
    _alpha: float
    _epsilon: float
    _weight_decay: float
    _momentum: float
    _centered: bool
    _v: List[NDArrayFloat]
    _b: List[NDArrayFloat]
    _average_g: List[NDArrayFloat]

    def __init__(
            self,
            parameters: Tuple[Tensor, ...],
            learning_rate: float = 1e-2,
            alpha: float = 0.99,
            epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            momentum: float = 0.0,
            centered: bool = False
            ) -> None:
        self._parameters = tuple(parameter.computational_node
                                 for parameter in parameters)

        self._learning_rate = learning_rate
        self._alpha = alpha
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._momentum = momentum
        self._centered = centered

        self._v = [np.zeros_like(parameter.gradient)
                   for parameter in self._parameters]
        self._b = [np.zeros_like(parameter.gradient)
                   for parameter in self._parameters]
        self._average_g = [np.zeros_like(parameter.gradient)
                           for parameter in self._parameters]

    def step(self, /) -> None:
        for i, parameter in enumerate(self._parameters):
            g = parameter.gradient.copy()

            if self._weight_decay != 0.0:
                g += self._weight_decay*parameter.result

            self._v[i] = self._alpha*self._v[i] + (1.0 - self._alpha)*g**2.0
            v_hat = self._v[i]

            if self._centered:
                self._average_g[i] = self._alpha*self._average_g[i] \
                    + (1.0 - self._alpha)*g
                v_hat -= self._average_g[i]**2.0

            if self._momentum > 0.0:
                self._b[i] = self._momentum*self._b[i] \
                    + g/(np.sqrt(v_hat) + self._epsilon)
                parameter.result -= self._learning_rate*self._b[i]
            else:
                parameter.result -= self._learning_rate*g \
                    / (np.sqrt(v_hat) + self._epsilon)