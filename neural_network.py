import numpy as np
from typing import Union


def relu(n: float) -> float:
    return max(0, n)


class Model: ...


class Layer:
    def __init__(
        self, cnt: int, learning_rate: float, child_layer: Union["Layer", "InputLayer"]
    ):
        self.child_layer = child_layer
        self.cnt = cnt
        self.neurons = [
            Neuron(bias, learning_rate, child_layer)
            for bias in np.random.randn(cnt)
        ]

    def learn(self, self_act_derivatives: np.ndarray):
        """해당 층 뉴런들 학습"""
        child_act_derivatives = np.zeros((self.child_layer.cnt,))
        for n, act_derivative in zip(self.neurons, self_act_derivatives):
            child_act_derivatives += n.learn(act_derivative)
        if isinstance(self.child_layer, Layer):
            self.child_layer.learn(child_act_derivatives)

    def get_activations(self) -> np.ndarray:
        child_activations = self.child_layer.get_activations()
        activations = np.array(
            list(map(lambda x: x.get_result(child_activations), self.neurons))
        )
        return activations


class InputLayer:
    def __init__(self, cnt: int):
        self.cnt = cnt
        self.activations = np.zeros((cnt,))

    def get_activations(self) -> np.ndarray:
        """활성도 벡터"""
        return self.activations

    def set_activations(self, data: np.ndarray):
        """입력"""
        self.activations = data


class Neuron:
    def __init__(
        self, bias: float, learning_rate: float, child_layer: Layer | InputLayer
    ):
        self.lr = learning_rate
        self.child_layer: Layer

        self.bias = bias
        self.child_layer = child_layer
        self.weights = np.abs(np.random.randn(child_layer.cnt)) * np.sqrt(2 / child_layer.cnt)

    def get_result(self, child_activations: np.ndarray):
        """자식 층에서 가중치에 활성도를 곱한 값을 가져와 편향을 더합니다."""
        self.child_activations = child_activations
        return relu(sum(child_activations * self.weights + self.bias))

    def learn(self, self_act_derivative: float):
        """학습"""
        activations = self.child_activations
        z = sum(activations * self.weights + self.bias)

        act_z_de = 1 if z > 0 else 0

        bias_derivative = 1 * act_z_de * self_act_derivative
        weight_derivative = activations * bias_derivative  # 뒤쪽 식이 똑같기 때문
        child_act_derivative = self.weights * bias_derivative

        self.bias -= self.lr * bias_derivative
        self.weights -= self.lr * weight_derivative
        # print(self.bias, self.weights)
        return child_act_derivative
