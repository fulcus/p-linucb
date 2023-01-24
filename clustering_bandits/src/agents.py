from abc import ABC, abstractmethod
import numpy as np
from random import Random


class Agent(ABC):
    def __init__(self, n_arms, random_state=1):
        self.n_arms = n_arms
        self.t = 0
        self.a_hist = []  # necessary?
        self.last_pull = None
        np.random.seed(random_state)
        self.randgen = Random(random_state)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, X, *args, **kwargs):
        pass

    def reset(self):
        self.t = 0
        self.last_pull = None


class UCB1Agent(Agent):
    def __init__(self, n_arms, max_reward=1):
        super().__init__(n_arms)
        self.max_reward = max_reward
        self.reset()

    def reset(self):
        super().reset()
        self.avg_reward = np.zeros(self.n_arms)
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a] + self.max_reward *
                np.sqrt(2 * np.log(self.t) / self.n_pulls[a]) for a in range(self.n_arms)]
        self.last_pull = np.argmax(ucb1)
        self.n_pulls[self.last_pull] += 1
        self.a_hist.append(self.last_pull)
        return self.last_pull

    def update(self, reward):
        self.rewards[self.last_pull].append(reward)
        self.avg_reward[self.last_pull] = (
            self.avg_reward[self.last_pull] * self.n_pulls[self.last_pull] + reward) / (self.n_pulls[self.last_pull] + 1)
        self.t += 1


class Clairvoyant(Agent):
    """Agent that knows the expected reward vector, and therefore always pull the optimal arm"""

    def __init__(self, n_arms, exp_reward, random_state=1):
        super().__init__(n_arms, random_state)
        self.exp_reward = exp_reward
        self.reset()

    def reset(self):
        super().reset()
        self.rewards = [[] for _ in range(self.n_arms)]
        return self

    def pull_arm(self):
        self.last_pull = np.argmax(self.exp_reward)
        self.a_hist.append(self.last_pull)
        return self.last_pull

    def update(self, reward):
        self.rewards[self.last_pull].append(reward)
        self.t += 1
