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
        obs_reward = self.exp_reward[action] + self.noise[self.t]
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def reset(self, i=0):
        self.t = 0
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.noise_std, self.n_rounds)
        return self


class LinearEnvironment:
    def __init__(self, n_rounds, theta, noise_std=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.theta = theta
        self.noise_std = noise_std
        self.random_state = random_state
        self.t = None
        self.noise = None
        self.rewards = None
        self.reset(0)

    def round(self, action):
        obs_reward = np.dot(self.theta, action) + self.noise[self.t]
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def reset(self, i=0):
        self.t = 0
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.noise_std, self.n_rounds)
        return self
