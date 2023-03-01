import argparse
import json
import os
import numpy as np
import itertools
import math


def onehot_arms(dim_arm):
    return np.eye(dim_arm).tolist()


def random_arms(n_arms, dim_arm, seed):
    np.random.seed(seed)
    return np.random.randint(0, 100, (n_arms, dim_arm)).tolist()


def vertex_arms(dim_arm, arm_norm):
    tuples = [(-arm_norm, arm_norm) for _ in range(dim_arm)]
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

    n_contexts = 10
    arm_dim = 4
    ctx_dim = 3
    # psi_dim = arm_dim * ctx_dim
    psi_dim = 2

    if args.arm == 'v':
        arm_coord_norm = 1
        arms = vertex_arms(arm_dim, arm_coord_norm)
        n_arms = len(arms)
    elif args.arm == 'o':
        arms = onehot_arms(arm_dim)
        n_arms = len(arms)
    elif args.arm == 'r':
        n_arms = 5
        arms = random_arms(n_arms, arm_dim, args.seed)

    
    # max local norm < 3.5
    # max_arm_norm = math.ceil(np.linalg.norm(np.outer([1, 1, 1, 1], [1, 1, 1])))
    # max global norm == 2 == np.linalg.norm([1, 1, 1, 1])
    
    max_arm_norm = math.ceil(np.max([np.linalg.norm(a) for a in arms]))

    np.random.seed(args.seed)
    theta = np.random.uniform(-3.0, 3.0, size=(1, psi_dim)).round(2).tolist()
    np.random.seed(args.seed)
    theta_p = np.random.uniform(-1.0, 1.0, size=(n_contexts, psi_dim))
    theta_p = theta_p.round(2).tolist()

    # only valid for v arms
    max_theta_norm_shared = math.ceil(math.sqrt(arm_dim) * 3)
    max_theta_norm_p = math.ceil(math.sqrt(arm_dim) * 1)
    max_theta_norm_sum = math.ceil(math.sqrt(arm_dim) * 4)

    np.random.seed(args.seed)
    context_set = np.random.uniform(0.0, 1.0, size=(n_contexts, ctx_dim))
    context_set = context_set.round(2).tolist()

    params = {
        "horizon": 5000,
        "n_epochs": 1,
        "sigma": 0.1,
        "seed": args.seed,
        "n_arms": n_arms,
        "psi_dim": psi_dim,
        "max_arm_norm": max_arm_norm,
        "max_theta_norm_sum": max_theta_norm_sum,
        "max_theta_norm_shared": max_theta_norm_shared,
        "max_theta_norm_p": max_theta_norm_p,
        "arms": arms,
        "theta": theta,
        "context_set": context_set,
        "theta_p": theta_p
    }

    filename = f"{args.arm}_{args.seed}.json"
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Generated {filename}")
