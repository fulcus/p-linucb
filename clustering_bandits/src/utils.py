import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_heatmap(fpath, arms, context_set, theta, theta_p):
    dict = {}
    for a_i, arm in enumerate(arms):
        dict[a_i] = []
        for c_i in range(len(context_set)):
            dict[a_i].append(round((theta + theta_p[c_i]) @ arm, 2))
    df = pd.DataFrame.from_dict(dict)
    sns.heatmap(df, annot=False)
    plt.savefig(fpath)
    # plt.show()
    # df.to_csv(out_dir + 'table.csv')
    # exit(0)


def vector_norm_bound(max_component_norm, dim):
    return np.sqrt(dim) * max_component_norm


def moving_average(arr, win=5):
    if len(arr) < win:
        return np.inf
    return np.mean(arr[-win:])
