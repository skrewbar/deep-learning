from neural_network import Layer, InputLayer, Neuron
import numpy as np

lr = 0.01
epochs = 20000

input_layer = InputLayer(2)
hidden_layer = Layer(2, lr, input_layer)
output_layer = Layer(1, lr, hidden_layer)

# input_layer.set_activations(np.array([1, 1]))
# answer = np.array([0])
# output = output_layer.get_activations()
# output_layer.learn((answer - output) ** 2)


inputs = list(map(lambda x: np.array(x), [[0, 0], [0, 1], [1, 0], [1, 1]]))
answers = [0, 1, 1, 0]

for i in range(1, epochs + 1):
    for k in range(4):
        input_layer.set_activations(inputs[k])
        output = output_layer.get_activations()
        error = sum((output - answers[k]) ** 2)
        error_der = 2 * (output - answers[k])
        output_layer.learn(error_der)

        if i % 10000 == 0:
            print(f"{i}째 | 출력: {output}, 답: {answers[k]}, 오차: {error}")
