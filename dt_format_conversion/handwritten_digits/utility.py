from handwritten_digits.constants import side, dir_name
from scipy.misc import imsave
import numpy as np
import os


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
