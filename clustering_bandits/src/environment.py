import logging
import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, n_rounds, arms, n_contexts, theta, context_distr="round_robin",
                 popular_freq=None, sigma=0.1, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.n_contexts = n_contexts
        self.sigma = sigma
        self.random_state = random_state
        self.context_distr = context_distr
        self.t = 0
        self.noise = None
        self.curr_index = None
        self.context_indexes = np.arange(self.n_contexts)
        self.rewards = []
        
        if popular_freq:
            long_tail_p = (1 - popular_freq) / (self.n_contexts - 1)
            self.p = [popular_freq] + [long_tail_p] * (self.n_contexts - 1)
            logging.log(logging.INFO, "context_distr: %s", self.p)

        self.reset(0)

    @abstractmethod
    def round(self, arm):
        self.t += 1

    def get_context(self):
        if self.context_distr == "uniform":
            self.curr_index = np.random.choice(
                self.context_indexes)
        elif self.context_distr == "long_tail":
            self.curr_index = np.random.choice(self.context_indexes, p=self.p)
        else:  # round robin
            self.curr_index = self.t % self.n_contexts
        return self.curr_index

    def reset(self, i=0):
        self.t = 0
        self.rewards = []
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.sigma,
                                      size=(self.n_rounds, self.n_contexts))
        return self


class PartitionedEnvironment(Environment):
    """exp_reward = theta * arm[:k] + theta_p[context] * arm[k:]"""

    def __init__(self, n_rounds, arms, n_contexts, theta, theta_p,
                 k, context_distr, popular_freq, sigma=0.01, random_state=1):
        super().__init__(n_rounds, arms, n_contexts,
                         theta, context_distr, popular_freq, sigma, random_state)
        self.theta_p = theta_p
        self.k = k  # first k global components

    def round(self, arm):
        obs_reward = (self.theta @ arm[:self.k]
                      + self.theta_p[self.curr_index] @ arm[self.k:]
                      + self.noise[self.t, self.curr_index])
        self.rewards.append(obs_reward)
        self.t += 1
        return obs_reward
