import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, n_rounds, arms, n_contexts, theta,
                 sampling_distr="round_robin", sigma=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.n_contexts = n_contexts
        self.sigma = sigma
        self.random_state = random_state
        self.sampling_distr = sampling_distr

        self.t = 0
        self.noise = None
        self.rewards = []
        self.context_indexes = np.arange(self.n_contexts)
        self.reset(0)

    @abstractmethod
    def round(self, arm, x_i):
        self.t += 1

    def get_context(self):
        if self.sampling_distr == "uniform":
            np.random.seed(self.random_state)
            self.curr_index = np.random.choice(
                self.context_indexes, size=1)
        elif self.sampling_distr == "long_tail":
            # np.random.seed(self.random_state + self.t)
            p = [0.5]
            long_tail_p = (1 - 0.5) / (self.n_contexts - 1)
            for i in range(1, self.n_contexts):
                p.append(long_tail_p)
            #print(p)
            #p = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            self.curr_index = np.random.choice(
                self.context_indexes, p=p)
        else:  # round robin
            self.curr_index = self.t % self.n_contexts
        return self.curr_index

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

    def round(self, arm):
        obs_reward = (self.theta @ arm[:self.k]
                      + self.theta_p[self.curr_index] @ arm[self.k:]
                      + self.noise[self.t, self.curr_index])
        self.rewards.append(obs_reward)
        self.t += 1
        return obs_reward
