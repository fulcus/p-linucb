from abc import ABC, abstractmethod
import numpy as np
from random import Random


class Agent(ABC):
    def __init__(self, n_arms, random_state=1):
        self.n_arms = n_arms
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        np.random.seed(random_state)
        self.randgen = Random(random_state)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, reward, *args, **kwargs):
        pass

    def reset(self):
        self.t = 0
        self.a_hist = []
        self.last_pull = None


class Clairvoyant(Agent):
    """Agent that knows the expected reward vector, and therefore always pull the optimal arm"""

    def __init__(self, n_arms, actions, exp_reward, random_state=1):
        super().__init__(n_arms, random_state)
        self.exp_reward = exp_reward
        self.idx_action = {i: a for i, a in enumerate(actions)}
        self.reset()

    def reset(self):
        super().reset()
        return self

    def pull_arm(self):
        i_a = np.argmax(self.exp_reward)
        self.last_pull = self.idx_action[i_a]
        self.a_hist.append(i_a)
        return self.last_pull

    def update(self, reward):
        self.t += 1


class UCB1Agent(Agent):
    def __init__(self, n_arms, actions, max_reward=1, random_state=1):
        super().__init__(n_arms, random_state)
        self.max_reward = max_reward
        self.idx_action = {i: a for i, a in enumerate(actions)}
        self.reset()

    def reset(self):
        super().reset()
        self.avg_reward = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a] + self.max_reward *
                np.sqrt(2 * np.log(self.t) / self.n_pulls[a]) for a in range(self.n_arms)]
        self.idx_last_pull = np.argmax(ucb1)
        self.n_pulls[self.idx_last_pull] += 1
        self.last_pull = self.idx_action[self.idx_last_pull]
        self.a_hist.append(self.idx_last_pull)
        return self.last_pull

    def update(self, reward):
        self.avg_reward[self.idx_last_pull] = (
            self.avg_reward[self.idx_last_pull] * self.n_pulls[self.idx_last_pull] + reward) / (self.n_pulls[self.idx_last_pull] + 1)
        self.t += 1


class LinUCBAgent(Agent):
    def __init__(self, actions, horizon, lmbd,
                 max_theta_norm, max_action_norm, random_state=1):
        assert lmbd > 0
        self.n_arms = actions.shape[0]
        self.action_dim = actions.shape[1]
        super().__init__(n_arms=self.n_arms, random_state=random_state)
        self.actions = actions
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_action_norm = max_action_norm
        self.action_idx = {tuple(a): i for i, a in enumerate(actions)}
        self.reset()

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.action_dim)
        self.b_vect = np.zeros((self.action_dim, 1))
        self.hat_h_vect = np.zeros((self.action_dim, 1))
        self.first = True
        return self

    def pull_arm(self):
        if self.first:
            u_t = self.actions[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.first = False
        else:
            u_t, _ = self._estimate_linucb_action()
            self.last_pull = u_t.reshape(self.action_dim, 1)
        i_a = self.action_idx[tuple(self.last_pull.squeeze())]
        self.a_hist.append(i_a)
        return self.last_pull

    def update(self, reward):
        self.V_t = self.V_t + (self.last_pull @ self.last_pull.T)
        self.b_vect = self.b_vect + self.last_pull * reward
        self.hat_h_vect = np.linalg.inv(self.V_t) @ self.b_vect
        self.t += 1

    def _beta_t_fun_linucb(self):
        return self.max_theta_norm * np.sqrt(self.lmbd) + \
            np.sqrt(
            2 * np.log(self.horizon) + (
                self.action_dim * np.log(
                    (self.action_dim * self.lmbd +
                     self.horizon * (self.max_action_norm ** 2)
                     ) / (self.action_dim * self.lmbd)
                )
            )
        )

    def _estimate_linucb_action(self):
        bound = self._beta_t_fun_linucb()
        obj_vals = np.zeros(self.n_arms)
        for i, act_i in enumerate(self.actions):
            act_i = act_i.reshape(self.action_dim, 1)
            obj_vals[i] = self.hat_h_vect.T @ act_i + bound * np.sqrt(
                act_i.T @ np.linalg.inv(self.V_t) @ act_i)
        return self.actions[np.argmax(obj_vals), :], np.argmax(obj_vals)