import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.ma import average


def psi_lin(a, x):
    return a


def psi_hadamard(a, x):
    return np.multiply(a, x)


def psi_cartesian(a, x):
    return np.outer(a, x).reshape(-1)


def save_heatmap(fpath, arms, context_set, theta, theta_p):
    dict = {}
    for a_i, arm in enumerate(arms):
        dict[a_i] = []
        for c_i in range(len(context_set)):
            psi = arm  # non-contextual
            dict[a_i].append(round((theta + theta_p[c_i]) @ psi, 2))
    df = pd.DataFrame.from_dict(dict)
    sns.heatmap(df, annot=False)
    plt.savefig(fpath)
    # plt.show()
    # df.to_csv(out_dir + 'table.csv')
    # exit(0)

def moving_average(arr, win=5):
    if len(arr) < win:
        return np.inf
    return sum(arr[-win:]) / win
