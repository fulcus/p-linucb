import argparse
import json
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generator',
        description='generates test cases')
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-t', '--type', choices=['l', 'c', 'p'], default='p')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 1000)
    np.random.seed(args.seed)

    n_arms = 5
    n_contexts = 10

    dim_arm = n_arms
    arms = np.eye(n_arms)
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

    if args.type in ['c', 'p']:
        contexts = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
        params["contexts"] = contexts.tolist(),
    if args.type == 'p':
        theta_p = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
        params["theta_p"] = theta_p.tolist(),

    print(f"Generated testcase_{args.seed} of type {args.type}")
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + f"testcase_{args.seed}.json", "w") as f:
        json.dump(params, f, indent=4)
