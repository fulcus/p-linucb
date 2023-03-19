import argparse
import json
import os
import numpy as np
import itertools
import math
from src.utils import *


def onehot_arms(dim_arm):
    return np.eye(dim_arm).tolist()


def random_arms(n_arms, dim_arm, seed):
    np.random.seed(seed)
    return np.random.randint(0, 100, (n_arms, dim_arm)).tolist()


def vertex_arms(dim_arm, max_a):
    tuples = [(-max_a, max_a) for _ in range(dim_arm)]
    arms = [a for a in itertools.product(*tuples)]
    return arms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generator',
        description='generates test cases')
    parser.add_argument('-s', '--seed', type=int,
                        default=np.random.randint(0, 1000))
    parser.add_argument('-a', '--arm', choices=['v', 'o', 'r'], default='v')
    args = parser.parse_args()

    lmbd = 1
    n_contexts = 10
    arm_dim = 4
    k = 2  # global dim
    max_th = 100
    max_th_p = 100
    max_a = 1

    local_dim = arm_dim - k

    if args.arm == 'v':
        arms = vertex_arms(arm_dim, max_a)
        n_arms = len(arms)
    elif args.arm == 'o':
        arms = onehot_arms(arm_dim)
        n_arms = len(arms)
    elif args.arm == 'r':
        n_arms = 5
        arms = random_arms(n_arms, arm_dim, args.seed)

    np.random.seed(args.seed)
    theta = np.random.uniform(-max_th, max_th, size=(1, k)).round(2).tolist()
    np.random.seed(args.seed+1)
    theta_p = np.random.uniform(-max_th_p, max_th_p,
                                size=(n_contexts, local_dim))
    theta_p = theta_p.round(2).tolist()

    # only valid for v arms
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
    }

    filename = f"{args.arm}_{args.seed}.json"
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Generated {filename}")
