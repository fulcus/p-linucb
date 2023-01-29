import argparse
import os
import numpy as np
import json


def generate_linear_bandits(seed=0, n_arms=5):
    """exp_reward = theta * arm"""
    arms = np.eye(n_arms)
    theta = np.random.uniform(-1.0, 1.0, size=(1, n_arms))
    max_arm_norm = np.max([np.linalg.norm(a) for a in arms])
    max_theta_norm = np.linalg.norm(theta)
    params = {
        "seed": seed,
        "arms": arms.tolist(),
        "n_arms": arms.shape[1],
        "theta": theta.tolist(),
        "max_arm_norm": max_arm_norm,
        "max_theta_norm": max_theta_norm,
        "horizon": 1000,
        "n_epochs": 10,
        "noise_std": 0.01,
    }
    return params


def generate_product_bandits(seed=0, n_arms=5, n_contexts=10):
    """exp_reward = theta * psi(arm, context) + theta_p[context] * psi(arm, context)"""
    dim_arm = n_arms
    arms = np.eye(n_arms)
    contexts = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
    theta_p = np.random.uniform(-1.0, 1.0, size=(n_contexts, dim_arm))
    theta = np.random.uniform(-1.0, 1.0, size=(1, dim_arm))
    # TODO with psi later
    max_arm_norm = np.max([np.linalg.norm(a) for a in arms])
    max_theta_norm = np.linalg.norm(theta)
    params = {
        "seed": seed,
        "contexts": contexts.tolist(),
        "theta_p": theta_p.tolist(),
        "arms": arms.tolist(),
        "n_arms": arms.shape[1],
        "theta": theta.tolist(),
        "max_arm_norm": max_arm_norm,
        "max_theta_norm": max_theta_norm,
        "horizon": 1000,
        "n_epochs": 10,
        "noise_std": 0.01,
    }
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generator',
        description='generates test cases')
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-t', '--type', choices=['l', 'p'], default='p')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 1000)
    np.random.seed(args.seed)
    if args.type == 'l':
        params = generate_linear_bandits(args.seed, n_arms=5)
    else:
        params = generate_product_bandits(args.seed)

    print(f"Generated {args.type}_testcase_{args.seed}")
    out_folder = 'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + f"{args.type}_testcase_{args.seed}.json", "w") as f:
        json.dump(params, f, indent=4)
