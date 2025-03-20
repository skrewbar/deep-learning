import numpy as np
from neural_network import Layer, InputLayer

LR = 0.005
EPOCHS = 20000

input_layer = InputLayer(2)
hidden_layer = Layer(4, LR, input_layer)
output_layer = Layer(1, LR, hidden_layer)

inputs = list(
    map(lambda x: np.array(x).reshape(1, 2), [[0, 0], [0, 1], [1, 0], [1, 1]])
)
answers = [0, 1, 1, 0]

for i in range(1, EPOCHS + 1):
    for k in range(4):
        input_layer.activations = inputs[k]
        output = output_layer.forward_propagation()
        error = sum((output - answers[k]) ** 2)
        error_grad = 2 * (output - answers[k])
        output_layer.back_propagation(error_grad)

        if i % 5000 == 0:
            print(f"{i}째 | 출력: {output}, 답: {answers[k]}, 오차: {error}")
