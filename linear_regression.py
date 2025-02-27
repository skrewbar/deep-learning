import numpy as np

w: float = 3
b: float = 76

x = np.array([2, 4, 6, 8])
y = np.array([4, 8, 12, 16])

COUNT = len(x)

LR = 0.03
EPOCHS = 2001

for i in range(EPOCHS):
    y_pred = w * x + b
    error = y - y_pred

    w_diff = (2 / COUNT) * sum(-x * error)
    b_diff = (2 / COUNT) * sum(-error)

    w -= LR * w_diff
    b -= LR * b_diff

    if i % 100 == 0:
        print(f"epoch={i}, 가중치={w}, 편향={b}, 평균오차={sum(error)/COUNT}")
