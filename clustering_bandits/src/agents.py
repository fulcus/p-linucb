from abc import ABC, abstractmethod
import numpy as np
from random import Random


class Agent(ABC):
    def __init__(self, arms, random_state=1):
        self.arms = arms
        self.n_arms = arms.shape[0]
        self.arm_dim = arms.shape[1]
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
    """Agent that knows the expected reward vector, 
    and therefore always pull the optimal arm"""

    def __init__(self, contexts, arms, theta, theta_p, psi=None, random_state=1):
        super().__init__(arms, random_state)
        self.contexts = contexts
        self.theta = theta
        self.theta_p = theta_p
        self.idx_arm = {i: a for i, a in enumerate(arms)}
        if psi is None:
            self.psi = lambda a, x: a
        else:
            self.psi = psi
        self.reset()

    def reset(self):
        super().reset()
        return self

    def pull_arm(self, context_i=None):
        exp_rewards = np.zeros(self.n_arms)
        if context_i is None:
            for i, arm in enumerate(self.arms):
                exp_rewards[i] = self.theta @ self.psi(arm, context)
        else:
            context = self.contexts[context_i]
            for i, arm in enumerate(self.arms):
                exp_rewards[i] = (self.theta @ self.psi(arm, context)
                                  + self.theta_p[context_i, :] @ self.psi(arm, context))
        i_a = np.argmax(exp_rewards)
        self.last_pull = self.idx_arm[i_a]
        self.a_hist.append(i_a)
        return self.last_pull

    def update(self, reward):
        self.t += 1


class UCB1Agent(Agent):
    def __init__(self, arms, max_reward=1, random_state=1):
        super().__init__(arms, random_state)
        self.max_reward = max_reward
        self.idx_arm = {i: a for i, a in enumerate(arms)}
        self.reset()

    def reset(self):
        super().reset()
        self.avg_reward = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a] + self.max_reward *
                np.sqrt(2 * np.log(self.t)
                / self.n_pulls[a]) for a in range(self.n_arms)]
        self.idx_last_pull = np.argmax(ucb1)
        self.n_pulls[self.idx_last_pull] += 1
        self.last_pull = self.idx_arm[self.idx_last_pull]
        self.a_hist.append(self.idx_last_pull)
        return self.last_pull

    def update(self, reward):
        self.avg_reward[self.idx_last_pull] = ((
            self.avg_reward[self.idx_last_pull]
            * self.n_pulls[self.idx_last_pull] + reward)
            / (self.n_pulls[self.idx_last_pull] + 1))
        self.t += 1


class LinUCBAgent(Agent):
    def __init__(self, arms, horizon, lmbd,
                 max_theta_norm, max_arm_norm, random_state=1):
        super().__init__(arms, random_state)
        assert lmbd > 0
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.arm_idx = {tuple(a): i for i, a in enumerate(arms)}
        self.reset()

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.arm_dim)
        self.b_vect = np.zeros((self.arm_dim, 1))
        self.theta_hat = np.zeros((self.arm_dim, 1))
        self.last_ucb = np.zeros(self.n_arms)
        self.first = True
        return self

    def pull_arm(self, arm=None):
        if arm is not None:
            self.last_pull = arm
            i_a = self.arm_idx[tuple(self.last_pull.squeeze())]
            self.a_hist.append(i_a)
            return self.last_pull

        if self.first:
            u_t = self.arms[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = u_t.reshape(self.arm_dim, 1)
            self.first = False
        else:
            u_t, _ = self._estimate_linucb_arm()
            self.last_pull = u_t.reshape(self.arm_dim, 1)
        i_a = self.arm_idx[tuple(self.last_pull.squeeze())]
        self.a_hist.append(i_a)
        return self.last_pull

    def update(self, reward):
        self.V_t = self.V_t + (self.last_pull @ self.last_pull.T)
        self.b_vect = self.b_vect + self.last_pull * reward
        self.theta_hat = np.linalg.inv(self.V_t) @ self.b_vect
        self.t += 1

    def _estimate_linucb_arm(self):
        bound = self._beta_t_fun_linucb()
        # self.last_ucb = np.zeros(self.n_arms)
        for i, act_i in enumerate(self.arms):
            act_i = act_i.reshape(self.arm_dim, 1)
            self.last_ucb[i] = self.theta_hat.T @ act_i + bound * np.sqrt(
                act_i.T @ np.linalg.inv(self.V_t) @ act_i)
        return self.arms[np.argmax(self.last_ucb), :], np.argmax(self.last_ucb)

    def _beta_t_fun_linucb(self):
        return (self.max_theta_norm * np.sqrt(self.lmbd)
                + np.sqrt(2 * np.log(self.horizon)
                + (self.arm_dim * np.log(
                    (self.arm_dim * self.lmbd
                     + self.horizon * (self.max_arm_norm ** 2))
                    / (self.arm_dim * self.lmbd)
                ))))


class ProductLinUCBAgent(Agent):
    """Combines a global linear bandit and an independent instance per context"""

    def __init__(self, contexts, arms, horizon, lmbd,
                 max_theta_norm, max_arm_norm, random_state=1):
        super().__init__(arms, random_state)
        self.random_state = random_state
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.arm_idx = {tuple(a): i for i, a in enumerate(arms)}
        self.contexts = contexts
        self.reset()

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm + theta_p * arm)"""
        self.last_context_i = context_i
        ucb = self.agent_global.last_ucb + \
            self.context_agent[context_i].last_ucb
        best_arm = self.arms[np.argmax(ucb), :]
        self.agent_global.pull_arm(best_arm)
        self.context_agent[context_i].pull_arm(best_arm)
        self.a_hist.append(best_arm)
        return best_arm

    def update(self, reward):
        pred_reward = self.agent_global.theta_hat.T @ self.agent_global.last_pull
        residual = reward - pred_reward
        self.context_agent[self.last_context_i].update(residual)
        self.agent_global.update(reward)

    def reset(self):
        super().reset()
        self.agent_global = LinUCBAgent(
            self.arms, self.horizon, self.lmbd,
            self.max_theta_norm, self.max_arm_norm, self.random_state)
        self.context_agent = [
            LinUCBAgent(
                self.arms, self.horizon,
                self.lmbd, self.max_theta_norm,
                self.max_arm_norm, self.random_state)
            for _ in self.contexts
        ]
