import argparse
import json
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generator',
        description='generates test cases')
    parser.add_argument('-s', '--seed', type=int, default=np.random.randint(0, 1000))
    parser.add_argument('-e', '--env', choices=['l', 'c', 'p'], default='p')
    args = parser.parse_args()
    np.random.seed(args.seed)

    n_arms = 5
    n_contexts = 10

    dim_arm = n_arms
    # arms = np.random.randint(0, 101, (n_arms, dim_arm))
    arms = np.eye(dim_arm)
    theta = np.random.uniform(-1.0, 1.0, size=(1, dim_arm))
    max_arm_norm = np.max([np.linalg.norm(a) for a in arms])
    max_theta_norm = np.linalg.norm(theta)
    params = {
        "seed": args.seed,
        "arms": arms.tolist(),
        "n_arms": arms.shape[1],
        "theta": theta.tolist(),
        "max_arm_norm": max_arm_norm,
        "max_theta_norm": max_theta_norm,
        "horizon": 1000,
        "n_epochs": 10,
        "noise_std": 0.01,
    }

    if args.env in ['c', 'p']:
        context_set = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
        params["context_set"] = context_set.tolist(),
    if args.env == 'p':
        theta_p = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
        params["theta_p"] = theta_p.tolist(),

    print(f"Generated testcase_{args.seed} of env {args.env}")
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + f"testcase_{args.env}_{args.seed}.json", "w") as f:
        json.dump(params, f, indent=4)
