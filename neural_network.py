import numpy as np
from typing import Self


def leaky_relu(n: float) -> float:
    return n if n > 0 else 0.01 * n


def leaky_relu_grad(n: float) -> float:
    return 1 if n > 0 else 0.01


vectorized_relu = np.vectorize(leaky_relu)
vectorized_relu_grad = np.vectorize(leaky_relu_grad)


class InputLayer:
    def __init__(self, cnt: int):
        self.cnt = cnt
        self.activations = np.zeros((1, cnt))

    def forward_propagation(self):
        return self.activations


class Layer:
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

    def forward_propagation(self) -> np.ndarray:
        self.prev_activation = self.child_layer.forward_propagation()
        self.weighted_sum = self.prev_activation.dot(self.weight) + self.bias.T
        return vectorized_relu(self.weighted_sum)

    def back_propagation(self, activation_grad: np.ndarray):
        bias_grad = vectorized_relu_grad(self.weighted_sum).T * activation_grad.T
        weight_grad = self.prev_activation.T.dot(bias_grad.T)

        self.bias -= self.lr * bias_grad
        self.weight -= self.lr * weight_grad

        if isinstance(self.child_layer, Layer):
            prev_activation_grad = self.weight.dot(bias_grad).T
            self.child_layer.back_propagation(prev_activation_grad)
