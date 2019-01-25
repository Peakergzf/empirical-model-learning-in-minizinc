from handwritten_digits.constants import m
from handwritten_digits.read_file import read_weight, read_training_set
from handwritten_digits.utility import generate_image, delete_images, print_digit, lst_to_mat
from handwritten_digits.ann import ann


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
