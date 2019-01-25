from handwritten_digits.constants import n, m, hidden_layer_size, output_layer_size

import scipy.io

weights_file = "data/ann_weights.mat"
digits_file = "data/digits.mat"


def read_weight():
    weight_mat = scipy.io.loadmat(weights_file)

    theta1_np = weight_mat["Theta1"]
    theta1 = [list(row) for row in theta1_np]

    theta2_np = weight_mat["Theta2"]
    theta2 = [list(row) for row in theta2_np]

    assert len(theta1) == hidden_layer_size
    for row in theta1:
        assert len(row) == n + 1

    assert len(theta2) == output_layer_size
    for row in theta2:
        assert len(row) == hidden_layer_size + 1

    return theta1, theta2


def read_training_set():
    digit_mat = scipy.io.loadmat(digits_file)

    x_np = digit_mat['X']

    x = [list(row) for row in x_np]
    assert len(x) == m
    for row in x:
        assert len(row) == n

    y = [lst[0] for lst in digit_mat['y']]
    assert len(y) == m
    # y: 500 10s, 500 1s, 500 2s, ... , 500 9s
    for i in range(m):
        assert y[i] == 10 if i < (m // 10) else y[i] == i // (m // 10)

    return x, y
