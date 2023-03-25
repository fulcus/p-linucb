import argparse
import json
import os
import numpy as np
import itertools
from src.utils import *


def vector_norm_bound(max_component, dim):
    return np.sqrt(dim) * max_component


def onehot_arms(dim_arm):
    return np.eye(dim_arm).tolist()


def vertex_arms(dim_arm, max_a):
    tuples = [(-max_a, max_a) for _ in range(dim_arm)]
    arms = [a for a in itertools.product(*tuples)]
    return arms


def var_len():
    lmbd = 1
    n_contexts = 10
    k = 2  # global dim
    max_th = 3
    max_th_p = 3
    max_a = 3

    # arms = vertex_arms(arm_dim, max_a)
    # n_arms = len(arms)
    n_arms_glob = 3
    n_arms_loc = 2
    n_arms = n_arms_glob * n_arms_loc
    arms_len = []
    theta_p = []

    global_arms = np.random.randint(0, max_a, size=(n_arms_glob, k)).tolist()
    # print(f"{global_arms=}")
    arms = []
    for _ in range(n_contexts):
        a_len = np.random.randint(k+1, 6)
        loc_dim = a_len - k
        local_arms = np.random.randint(
            0, max_a, size=(n_arms_loc, loc_dim)).tolist()
       # print(f"{i} {local_arms=}")
        arms_agent = []
        for g_arm in global_arms:
            for l_arm in local_arms:
                arms_agent.append(g_arm + l_arm)

        arms.append(arms_agent)

        theta_p_i = np.random.randint(-max_th_p, max_th_p,
                                      size=loc_dim).tolist()
        arms_len.append(a_len)
        theta_p.append(theta_p_i)

    # print("arms=", arms)
    theta = np.random.randint(-max_th, max_th, size=(1, k)).round(2).tolist()

    arm_dim = max(arms_len)
    local_dim = arm_dim - k

    max_theta_norm = vector_norm_bound(max_th, arm_dim)
    max_theta_norm_local = vector_norm_bound(max_th, local_dim)
    max_arm_norm = vector_norm_bound(max_a, arm_dim)
    max_arm_norm_local = vector_norm_bound(max_a, local_dim)

    params = {
        "horizon": 5000,
        "n_epochs": 1,
        "sigma": 0.1,
        "seed": args.seed,
        "n_arms": n_arms,
        "n_contexts": n_contexts,
        "k": k,
        "lmbd": lmbd,
        "max_arm_norm": max_arm_norm,
        "max_arm_norm_local": max_arm_norm_local,
        "max_theta_norm": max_theta_norm,
        "max_theta_norm_local": max_theta_norm_local,
        "theta": theta,
        "theta_p": theta_p,
        "arms": arms,
        "arms_len": arms_len
    }

    filename = f"diff_len_{args.seed}.json"
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Generated {filename}")


def same_len():
    lmbd = 1
    n_contexts = 10
    k = 2  # global dim
    max_th = 3
    max_th_p = 3
    max_a = 3

    n_arms_glob = 3
    n_arms_loc = 2
    n_arms = n_arms_glob * n_arms_loc
    arm_dim = 5
    loc_dim = arm_dim - k
    theta_p = np.random.randint(-max_th_p, max_th_p,
                                size=(n_contexts, loc_dim)).tolist()
    arms_agent = vertex_arms(arm_dim, max_a)
    arms = [arms_agent for _ in range(n_contexts)]

    theta = np.random.randint(-max_th, max_th, size=(1, k)).round(2).tolist()

    max_theta_norm = vector_norm_bound(max_th, arm_dim)
    max_theta_norm_local = vector_norm_bound(max_th, loc_dim)
    max_arm_norm = vector_norm_bound(max_a, arm_dim)
    max_arm_norm_local = vector_norm_bound(max_a, loc_dim)

    params = {
        "horizon": 5000,
        "n_epochs": 1,
        "sigma": 0.1,
        "seed": args.seed,
        "n_arms": n_arms,
        "n_contexts": n_contexts,
        "k": k,
        "lmbd": lmbd,
        "max_arm_norm": max_arm_norm,
        "max_arm_norm_local": max_arm_norm_local,
        "max_theta_norm": max_theta_norm,
        "max_theta_norm_local": max_theta_norm_local,
        "theta": theta,
        "theta_p": theta_p,
        "arms": arms,
    }

    filename = f"same_len_{args.seed}.json"
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Generated {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generator',
        description='generates test cases')
    parser.add_argument('-s', '--seed', type=int,
                        default=np.random.randint(0, 1000))
    args = parser.parse_args()
    same_len()
