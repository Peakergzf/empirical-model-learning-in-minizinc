from handwritten_digits.python_implementation.read_file import read_training_set, read_weight


constants = "n = 400;\nhidden_layer_size = 25;\noutput_layer_size = 10;\n\n"


def get_weight(theta1, theta2):
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


def get_x(x):
    res = "x = "
    res += str(x) + ";\n\n"
    return res


def main():
    with open("data.dzn", 'w'):
        pass

    with open("data.dzn", 'a') as data_file:
        data_file.write(constants)

    with open("data.dzn", 'a') as data_file:
        data_file.write(get_weight(*read_weight()))

    with open("data.dzn", 'a') as data_file:
        # access x list, then access the first x
        data_file.write(get_x(read_training_set()[0][0]))


if __name__ == '__main__':
    main()
