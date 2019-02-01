import pandas as pd
import math

# wdp_bal constants
n = 288  # the number of jobs
m = 48  # the number of cores

# ann constants
X_SIZE = 4  # input layer size
A_SIZE = 2  # hidden layer size

# file names
ANN_FILE = "data/ann1.txt"
NEIGH_FILE = "data/neigh.txt"
SOL_FILE = "data/sol.txt"
CPI_FILE = "data/cpi.txt"


def read_weights():
    ann_raw = pd.read_csv(ANN_FILE, sep=",", header=None)

    # remove the first three columns
    ann_df = ann_raw.iloc[:, 3:]

    # split theta1 and theta2
    theta1 = ann_df.iloc[:, :A_SIZE * (X_SIZE + 1)]
    theta2 = ann_df.iloc[:, A_SIZE * (X_SIZE + 1):]

    # convert to python list
    theta1_2d = theta1.values.tolist()
    theta2_2d = theta2.values.tolist()

    # one ann for each core
    assert len(theta1_2d) == m
    assert len(theta2_2d) == m

    for theta1_1d in theta1_2d:
        assert len(theta1_1d) == A_SIZE * (X_SIZE + 1)
    for theta2_1d in theta2_2d:
        assert len(theta2_1d) == A_SIZE + 1

    theta1_3d = [[row[:X_SIZE + 1], row[X_SIZE + 1:]] for row in theta1_2d]

    return theta1_3d, theta2_2d


def read_neigh():
    with open(NEIGH_FILE, 'r') as f:
        lines = f.readlines()

    input_neighs = [line.split(',')[0].split(': ')[1].split() for line in lines]
    assert len(input_neighs) == m

    return [list(map(int, neigh)) for neigh in input_neighs]


def read_sol():
    with open(SOL_FILE, 'r') as f:
        line = f.readlines()

    sol_lst = line[0].split(", ")
    assert len(sol_lst) == n

    return list(map(lambda x: int(x.split("->")[1]), sol_lst))


def read_cpi():
    with open(CPI_FILE, 'r') as f:
        line = f.readlines()

    cpi_lst = line[0].rstrip().split(',')
    assert len(cpi_lst) == n

    return list(map(float, cpi_lst))


def ann(x, theta1, theta2):
    """
    forward propagation
    :param x: a list of input features
    :param theta1: weights from input layer to hidden layer
    :param theta2: weights from hidden layer to hypothesis
    :return: hypothesis
    """
    assert len(x) == X_SIZE
    assert len(theta1) == A_SIZE
    for row in theta1:
        assert len(row) == X_SIZE + 1
    assert len(theta2) == A_SIZE + 1

    x.append(1.0)  # bias unit
    a = [0.0 for _ in range(A_SIZE)]

    # from input layer to hidden layer
    for i in range(A_SIZE):
        x_theta1 = 0.0
        for j in range(X_SIZE + 1):
            x_theta1 += x[j] * theta1[i][j]
        a[i] = math.tanh(x_theta1)

    a.append(1.0)  # bias unit

    # from hidden layer to hypothesis
    a_theta2 = 0.0
    for i in range(A_SIZE + 1):
        a_theta2 += a[i] * theta2[i]
    h = math.tanh(a_theta2)

    return h


def main():
    # cpi values for each job
    wld = read_cpi()

    # ================= input scaling ======================
    cpi_vmin = 0
    cpi_vmax = 35

    cpi_n_sub = 0.5 * (cpi_vmax - cpi_vmin)
    cpi_n_div = 0.5 * (cpi_vmax - cpi_vmin)

    norm_cpi = [(v - cpi_n_sub) / cpi_n_div for v in wld]

    # cpi_lst = [int(round(v * 10000)) for v in norm_cpi]

    cpi_lst = norm_cpi

    # ==================================================

    # neighbor core index for each core
    neighs = read_neigh()

    # mapped core index for each job
    sol = read_sol()

    # indexes of the jobs on each core
    jobs_idx = [[] for _ in range(m)]
    for job in range(n):
        jobs_idx[sol[job]].append(job)

    # list of cpi of the jobs on each core
    jobs_cpi = [[cpi_lst[idx] for idx in jobs_idx[k]] for k in range(m)]

    # average cpi of the jobs on each core

    avg_cpi = [sum(jobs_cpi[k]) / float(n / m) for k in range(m)]

    # minimum cpi of the jobs on each core
    min_cpi = [min(jobs_cpi[k]) for k in range(m)]

    # average of avg_cpi of the neighbors for each core
    neigh_cpi = [sum([avg_cpi[j] for j in neighs[k]]) / float(len(neighs[k])) for k in range(m)]

    # average of avg_cpi of all the other cores except for neighbours and itself
    other_cpi = [sum([avg_cpi[j] for j in range(m) if j not in neighs[k] and j != k]) / float(m - len(neighs[k]) - 1)
                 for k in range(m)]

    # weights used in ann
    theta1, theta2 = read_weights()

    # use ann to determine the efficiency for each core
    eff_s = [ann([avg_cpi[k], min_cpi[k], neigh_cpi[k], other_cpi[k]], theta1[k], theta2[k]) for k in range(m)]

    # ================= output scaling ======================
    eff_vmin = 0.0
    eff_vmax = 1.0

    eff_n_sub = 0.5 * (eff_vmax + 0.2 - eff_vmin)
    eff_n_div = 0.5 * (eff_vmax + 0.2 - eff_vmin)

    # eff_s = [(e - eff_n_sub) / eff_n_div for e in eff]

    eff = [e * eff_n_div + eff_n_sub for e in eff_s]
    # ==================================================

    for k in range(m):
        print("core " + ('0' if k < 10 else '') + str(k) + ": " +
              "avg={:16.14f}, min={:16.14f}, neigh={:16.14f}, other={:16.14f}, eff={:16.14f}"
              .format(avg_cpi[k], min_cpi[k], neigh_cpi[k], other_cpi[k], eff[k]))

    print("obj={:16.14f}".format(min(eff)))


if __name__ == '__main__':
    main()
