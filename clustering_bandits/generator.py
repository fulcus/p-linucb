import argparse
import json
import os
import numpy as np
import itertools


def vector_norm_bound(max_component, dim):
    return np.sqrt(dim) * max_component


def onehot_arms(dim_arm):
    return np.eye(dim_arm).tolist()


def vertex_arms(dim_arm, max_a):
    tuples = [(-max_a, max_a) for _ in range(dim_arm)]
    arms = [a for a in itertools.product(*tuples)]
    return arms


def main(args):
    lmbd = 1
    n_contexts = 10
    k = 2  # global dim
    max_th = 3
    max_th_p = 3
    max_a = 3

    arm_dim_max = 5
    local_dim_max = arm_dim_max - k
    n_arms = 2 ** arm_dim_max  # vertices

    theta = np.random.randint(-max_th, max_th, size=(1, k)).round(2).tolist()
    if args.armlen == "difflen":
        theta_p = []
        arms = []
        for _ in range(n_contexts):
            arm_len = np.random.randint(k+1, arm_dim_max)
            loc_dim = arm_len - k
            arms_agent = vertex_arms(arm_len, max_a)
            arms.append(arms_agent)
            theta_p_i = np.random.randint(-max_th_p, max_th_p,
                                          size=loc_dim).tolist()
            theta_p.append(theta_p_i)
    else:
        theta_p = np.random.randint(-max_th_p, max_th_p,
                                    size=(n_contexts, local_dim_max)).tolist()
        arms_agent = vertex_arms(arm_dim_max, max_a)
        arms = [arms_agent for _ in range(n_contexts)]

    max_theta_norm = vector_norm_bound(max_th, arm_dim_max)
    max_theta_norm_local = vector_norm_bound(max_th, local_dim_max)
    max_arm_norm = vector_norm_bound(max_a, arm_dim_max)
    max_arm_norm_local = vector_norm_bound(max_a, local_dim_max)

    params = {
        "horizon": 5000,
        "n_epochs": 1,
        "sigma": 0.1,
        "seed": args.seed,
        "context_distr": args.context_distr,
        "popular_freq": None,
        "err_th": args.err_th,
        "max_th": max_th,
        "max_th_p": max_th_p,
        "max_a": max_a,
        "k": k,
        "arm_dim_max": arm_dim_max,
        "n_arms": n_arms,
        "n_contexts": n_contexts,
        "lmbd": lmbd,
        "max_arm_norm": max_arm_norm,
        "max_arm_norm_local": max_arm_norm_local,
        "max_theta_norm": max_theta_norm,
        "max_theta_norm_local": max_theta_norm_local,
        "theta": theta,
        "theta_p": theta_p,
        "arms": arms
    }

    filename = f"{args.armlen}_{args.context_distr}_{args.seed}.json"
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
    parser.add_argument('-a', '--armlen', type=str, choices=["difflen", "eqlen"],
                        default="eqlen")
    parser.add_argument('-c', '--context_distr', type=str, choices=["uniform", "long_tail", "round_robin"],
                        default="round_robin")
    parser.add_argument('-e', '--err_th', type=float, default=0.01)

    args = parser.parse_args()
    main(args)
