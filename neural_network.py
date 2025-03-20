from typing import Self
import numpy as np


def leaky_relu(n: np.ndarray) -> np.ndarray:
    """leaky_relu 함수"""
    return np.where(n > 0, n, 0.01 * n)


def leaky_relu_grad(n: np.ndarray) -> np.ndarray:
    """leaky_relu의 도함수"""
    return np.where(n > 0, 1, 0.01)


class InputLayer:
    """입력층"""

    def __init__(self, cnt: int):
        self.cnt = cnt
        self.activations = np.zeros((1, cnt))

    def forward_propagation(self):
        """순전파 활성도 반환"""
        return self.activations


class Layer:
    """뉴런 층"""

    def __init__(self, cnt: int, learning_rate: float, child_layer: Self | InputLayer):
        self.cnt = cnt
        self.lr = learning_rate
        self.child_layer = child_layer

        self.bias: np.ndarray
        self.weight: np.ndarray
        self.bias = np.zeros((cnt, 1))  # (현재층 X 1)
        self.weight = np.random.randn(child_layer.cnt, cnt) * np.sqrt(
            2 / child_layer.cnt
        )  # (이전층 X 현재층)
        self.activation_grad = np.zeros((1, cnt))

        self.weighted_sum: np.ndarray
        self.prev_activation: np.ndarray

    def forward_propagation(self) -> np.ndarray:
        """순전파 계산"""
        self.prev_activation = self.child_layer.forward_propagation()
        self.weighted_sum = self.prev_activation.dot(self.weight) + self.bias.T
        return leaky_relu(self.weighted_sum)

    def back_propagation(self, activation_grad: np.ndarray):
        """오차역전파"""
        bias_grad = leaky_relu_grad(self.weighted_sum).T * activation_grad.T
        weight_grad = self.prev_activation.T.dot(bias_grad.T)

        self.bias -= self.lr * bias_grad
        self.weight -= self.lr * weight_grad

        if isinstance(self.child_layer, Layer):
            prev_activation_grad = self.weight.dot(bias_grad).T
            self.child_layer.back_propagation(prev_activation_grad)
