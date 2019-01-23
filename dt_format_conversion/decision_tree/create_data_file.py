from decision_tree.dt_format_conversion import convert_tree_format

TREE_FILE = "ptf_feat3.txt"


def create_data_file(output):
    """
    read cpi values from the file and takes the output string,
    writes to the output file
    """
    with open("cpi.in") as f:
        cpi = f.read()

    with open("data.dzn", 'w'):
        pass

    with open("data.dzn", 'a') as data_file:
        data_file.write(cpi + output)


def main():
    output = convert_tree_format(TREE_FILE)
    create_data_file(output)


if __name__ == '__main__':
    main()
