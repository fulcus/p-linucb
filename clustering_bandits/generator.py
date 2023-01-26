import os
import numpy as np
import json
import sys


if __name__ == '__main__':
    out_folder = f'clustering_bandits/test/input/'
    os.makedirs(out_folder, exist_ok=True)
    # exp_reward = theta * actions
    seed = np.random.randint(0, 100)
    np.random.seed(seed)
    
    n_actions = 5
    actions = np.eye(n_actions)
    theta = np.random.uniform(-1.0, 1.0, size=(1, n_actions)) 
    exp_reward = np.dot(theta, actions)
    max_action_norm = 1 # standard basis
    max_theta_norm = np.linalg.norm(theta)
    params = {
    "seed": seed,
    "exp_reward": exp_reward.tolist(),
    "actions": actions.tolist(),
    "n_actions": actions.shape[1],
    "theta": theta.tolist(),
    "exp_reward": exp_reward.tolist(), 
    "max_action_norm": max_action_norm,
    "max_theta_norm": max_theta_norm,
    "horizon": 1000,
    "n_epochs": 10,
    "noise_std": 0.01,
    }

    out_file = open(out_folder + f"testcase_{seed}.json", "w")
    json.dump(params, out_file, indent=4)
    out_file.close()
