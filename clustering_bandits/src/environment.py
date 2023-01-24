import numpy as np


class Environment:
    def __init__(self, n_rounds, exp_reward, noise_std=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.noise_std = noise_std
        self.exp_reward = exp_reward
        self.random_state = random_state
        self.t = None
        self.noise = None
        self.rewards = None
        self.reset(0)

    def round(self, action):
        """Chosen action, appends generated reward to array X"""
        obs_reward = self.exp_reward[action] + self.noise[self.t]
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def reset(self, i=0):
        self.t = 0
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.noise_std, self.n_rounds)
        # p of benoulli
        #self.exp_reward = np.random.uniform(0, 1, n_arms) 
        return self
