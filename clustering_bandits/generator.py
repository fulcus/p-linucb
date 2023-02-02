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
    parser.add_argument('-e', '--env', choices=['l', 'c', 'p'], default='p')
    parser.add_argument('-a', '--arm', choices=['v', 'o', 'r'], default='v')

    args = parser.parse_args()

    n_contexts = 10
    dim_arm = 4
    arm_norm = 1

    if args.arm == 'v':
        arms = vertex_arms(dim_arm, arm_norm)
        n_arms = len(arms)
    elif args.arm == 'o':
        arms = onehot_arms(dim_arm)
        n_arms = len(arms)
    elif args.arm == 'r':
        n_arms = 5
        arms = random_arms(n_arms, dim_arm, args.seed)
        arm_norm = np.max([np.linalg.norm(a) for a in arms])

    np.random.seed()
    theta = np.random.uniform(-3.0, 3.0, size=(1, dim_arm)).round(2)
    max_theta_norm = math.ceil(np.linalg.norm(theta))
    params = {
        "horizon": 10000,
        "n_epochs": 10,
        "sigma": 0.1,
        "seed": args.seed,
        "n_arms": n_arms,
        "max_arm_norm": arm_norm,
        "max_theta_norm": max_theta_norm,
        "arms": arms,
        "theta": theta.tolist()
    }

    if args.env in ['c', 'p']:
        np.random.seed(args.seed)
        context_set = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
        params["context_set"] = context_set.round(2).tolist(),
    if args.env == 'p':
        np.random.seed(args.seed)
        theta_p = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
        params["theta_p"] = theta_p.round(2).tolist(),

    filename = f"testcase_{args.env}_{args.arm}_{args.seed}.json"
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Generated {filename}")
