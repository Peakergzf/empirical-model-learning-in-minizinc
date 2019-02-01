import scipy.io
from scipy.misc import imsave
import numpy as np
import os
import math

m = 5000  # number of training examples
n = 400  # number of features (input layer size)
side = 20  # the image is 20x20 pixel
hidden_layer_size = 25
output_layer_size = 10

dir_name = r"C:\Users\peake\empirical-model-learning-in-minizinc\handwritten-digits\python-implementation"
weights_file = "../data/ann_weights.mat"
digits_file = "../data/digits.mat"


# ============================================================================
#                                READ FILE
# ============================================================================

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


# ============================================================================
#                                ANN
# ============================================================================

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


# ============================================================================
#                                UTILITY
# ============================================================================

def lst_to_mat(lst):
    """
    convert from a 1d vector of size n to a 2d matrix of size sqrt(n) x sqrt(n)
    """
    mat = [[0.0 for _ in range(side)] for _ in range(side)]
    for i in range(side):
        for j in range(side):
            # construct column by column (MatLab reshape function)
            mat[i][j] = lst[j * side + i]
    return mat


def mat_to_lst(mat):
    """
    convert from 2d matrix of size n x n to a 1d vector of size n^2
    """
    lst = []
    for i in range(side):
        for j in range(side):
            lst.append(mat[j][i])
    return lst


def print_float_list(lst):
    res = "["
    for elem in lst:
        res += "{:.3f}, ".format(elem)
    res = res[:-2]
    res += "]"
    print(res)


def print_digit(x):
    """
    :param x: a list of pixel values
    print the digit in console
    """
    res = ""
    for i in range(side):
        for j in range(side):
            if abs(x[j * side + i]) > 0.1:
                res += ". "
            else:
                res += "  "
        res += "\n"
    print(res)


def generate_image(mat, image_name):
    """
    :param mat: a matrix of pixel values
    :param image_name:
    generate an image file
    """
    imsave(image_name, np.array(mat))


def delete_images():
    """
    delete all the images previously created
    """
    item_lst = os.listdir(dir_name)

    for item in item_lst:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_name, item))


# ============================================================================
#                                DATA FOR MZN
# ============================================================================

constants = "n = 400;\nhidden_layer_size = 25;\noutput_layer_size = 10;\n\n"


def get_weight(theta1, theta2):
    """
    :param theta1: 2d list of floats
    :param theta2: 2d list of floats
    :return: string used for MiniZinc model data file
    """
    res = "theta1 = array2d(1..hidden_layer_size, 1..n+1, ["
    for row in theta1:
        for elem in row:
            res += str(elem) + ', '
    res = res[:-2]
    res += "]);\n\ntheta2 = array2d(1..output_layer_size, 1..hidden_layer_size+1, ["

    for row in theta2:
        for elem in row:
            res += str(elem) + ', '
    res = res[:-2]
    res += "]);\n\n"

    return res


def get_input_feat(x):
    """
    :param x: a list of floats representing an image
    :return: string used for MiniZinc model data file
    """
    res = "original = "
    res += str(x) + ";\n\n"
    return res


def pre_process_data_for_mzn():
    with open("data.dzn", 'w'):
        pass

    with open("data.dzn", 'a') as data_file:
        data_file.write(constants)

    with open("data.dzn", 'a') as data_file:
        data_file.write(get_weight(*read_weight()))

    with open("data.dzn", 'a') as data_file:
        # access list x from x y pair, then access the first list in x (the first image)
        data_file.write(get_input_feat(read_training_set()[0][0]))


# ============================================================================
#                                MAIN
# ============================================================================


def main():
    delete_images()

    x, y = read_training_set()
    theta1, theta2 = read_weight()

    wrong_pred_cnt = 0  # wrong prediction count
    for i in range(m):
        # generate one image and print to console once for each digit (every m // 10 examples)
        if i % (m // 10) == 0:
            print_digit(x[i])
            generate_image(lst_to_mat(x[i]), "digit{}.png".format(str(i // (m // 10))))

        h = ann(x[i], theta1, theta2)

        if str(y[i])[-1] != str(h):
            wrong_pred_cnt += 1

    print("{} wrong predictions out of {} examples".format(wrong_pred_cnt, m))
    print("correctness rate: {:.3}".format(1 - wrong_pred_cnt / m))


if __name__ == '__main__':
    main()
