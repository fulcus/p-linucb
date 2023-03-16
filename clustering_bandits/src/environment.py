import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):

    def __init__(self, n_rounds, arms, context_set, theta, sigma=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.context_set = context_set
        self.sigma = sigma
        self.random_state = random_state

        self.t = None
        self.noise = None
        self.rewards = np.array([])
        self.n_contexts = self.context_set.shape[0]
        self.context_indexes = np.arange(self.n_contexts)
        self.reset(0)

    def round_all(self, pulled_arms_i):
        """computes reward for each context, arm pair
            pulled_arms_i:
        """
        obs_rewards = []
        for x_i, a_j in zip(self.curr_indexes, pulled_arms_i):
            obs_rewards.append(self.round(a_j, x_i))
        # logging sum of rewards
        self.rewards = np.append(self.rewards, sum(obs_rewards))
        self.t += 1
        return obs_rewards

    @abstractmethod
    def round(self, arm, x_i):
        pass

    def get_contexts(self):
        self.curr_indexes = self.context_indexes
        # could be generated by sampling at each round, as below
        # np.random.seed(self.random_state)
        # self.curr_indexes = np.random.choice(
        #     self.context_indexes, size=self.n_contexts)
        # p = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        # self.curr_indexes = np.random.choice(
        #     self.context_indexes, size=self.n_contexts, p=p)
        return self.curr_indexes

    def reset(self, i=0):
        self.t = 0
        self.rewards = np.array([])
        np.random.seed(self.random_state + i)
        # different noise for each context
        # TODO OR diff noise for each round
        self.noise = np.random.normal(0, self.sigma,
                                      size=(self.n_rounds, self.n_contexts))
        return self


class PartitionedEnvironment(Environment):
    """exp_reward = theta * arm[:k] + theta_p[context] * arm[k:]"""

    def __init__(self, n_rounds, arms, context_set, theta, theta_p, k, sigma=0.01, random_state=1):
        super().__init__(n_rounds, arms, context_set, theta, sigma, random_state)
        self.theta_p = theta_p
        self.k = k  # first k global components

    def round(self, arm, x_i):
        obs_reward = (self.theta @ arm[:self.k]
                      + self.theta_p[x_i] @ arm[self.k:]
                      + self.noise[self.t, x_i])
        return obs_reward
