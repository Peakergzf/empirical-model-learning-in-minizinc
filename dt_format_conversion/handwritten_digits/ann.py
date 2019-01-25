from handwritten_digits.constants import n, hidden_layer_size, output_layer_size
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ann(x, theta1, theta2):
    """
    computes forward propagation
     x
    a1 --- z2
           a2 --- z3
                  h
    """
    a1 = x
    a1.insert(0, 1.0)  # add bias unit

    z2 = [0.0 for _ in range(hidden_layer_size)]
    for i in range(hidden_layer_size):
        for j in range(n + 1):
            z2[i] += a1[j] * theta1[i][j]
    a2 = [sigmoid(z2i) for z2i in z2]

    a2.insert(0, 1.0)  # add bias unit

    z3 = [0.0 for _ in range(output_layer_size)]
    for i in range(output_layer_size):
        for j in range(hidden_layer_size + 1):
            z3[i] += a2[j] * theta2[i][j]
    h = [sigmoid(z3i) for z3i in z3]

    digit = h.index(max(h))

    # digit 0 is mapped to index 9 since 0 based index in MatLab
    return 0 if digit == 9 else digit + 1
