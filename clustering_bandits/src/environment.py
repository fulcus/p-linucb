import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, n_rounds, arms, n_contexts, theta,
                 sampling_distr=None, sigma=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.n_contexts = n_contexts
        self.sigma = sigma
        self.random_state = random_state
        self.sampling_distr = sampling_distr

        self.t = None
        self.noise = None
        self.rewards = []
        self.context_indexes = np.arange(self.n_contexts)
        self.reset(0)

    def round_all(self, pulled_arms):
        """computes reward for each context, arm pair"""
        obs_rewards = []
        for x_i, arm in zip(self.curr_indexes, pulled_arms):
            obs_rewards.append(self.round(arm, x_i))
        self.t += 1
        # logging sum of rewards
        self.rewards.append(sum(obs_rewards))
        # print(f"{self.t} roundall={sum(obs_rewards)}")
        return obs_rewards

    @abstractmethod
    def round(self, arm, x_i):
        pass

    def get_contexts(self):
        if self.sampling_distr == "uniform":
            np.random.seed(self.random_state)
            self.curr_indexes = np.random.choice(
                self.context_indexes, size=self.n_contexts)
        elif self.sampling_distr == "long_tail":
            np.random.seed(self.random_state)
            p = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            self.curr_indexes = np.random.choice(
                self.context_indexes, size=self.n_contexts, p=p)
        else:  # round robin
            self.curr_indexes = self.context_indexes
        return self.curr_indexes

    def reset(self, i=0):
        self.t = 0
        self.rewards = []
        np.random.seed(self.random_state + i)
        # different noise for each context
        # TODO OR diff noise for each round
        self.noise = np.random.normal(0, self.sigma,
                                      size=(self.n_rounds, self.n_contexts))
        return self


class PartitionedEnvironment(Environment):
    """exp_reward = theta * arm[:k] + theta_p[context] * arm[k:]"""

    def __init__(self, n_rounds, arms, n_contexts, theta, theta_p,
                 sampling_distr, k, sigma=0.01, random_state=1):
        super().__init__(n_rounds, arms, n_contexts,
                         theta, sampling_distr, sigma, random_state)
        self.theta_p = theta_p
        self.k = k  # first k global components

    def round(self, arm, c_i):
        # print("round, ctx=", c_i, arm, self.theta_p[c_i])
        # print("g",self.theta, arm[:self.k])
        # print(c_i, self.theta_p[c_i], arm[self.k:])

        obs_reward = (self.theta @ arm[:self.k]
                      + self.theta_p[c_i] @ arm[self.k:]
                      + self.noise[self.t, c_i])
        # print("env", self.t, c_i, arm, "real=",(self.theta @ arm[:self.k]
        #               + self.theta_p[c_i] @ arm[self.k:]).squeeze(), "obs=",obs_reward)

        return obs_reward
