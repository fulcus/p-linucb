import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error


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


def moving_mape(y_true, y_pred, win=5):
    if len(y_true) < win:
        return np.inf
    return mean_absolute_percentage_error(y_true[-win:], y_pred[-win:])
