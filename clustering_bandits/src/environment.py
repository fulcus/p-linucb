import numpy as np


class LinearEnvironment:
    """exp_reward = theta * arm"""

    def __init__(self, n_rounds, arms, theta, sigma=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.sigma = sigma
        self.random_state = random_state
        self.t = None
        self.noise = None
        self.rewards = np.array([])
        self.reset(0)

    def round(self, arm_i):
        obs_reward = self.theta @ self.arms[arm_i] + self.noise[self.t]
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def reset(self, i=0):
        self.t = 0
        self.rewards = np.array([])
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.sigma, self.n_rounds)
        return self


class ContextualLinearEnvironment(LinearEnvironment):
    """exp_reward = theta * psi(arm, context)"""

    def __init__(self, n_rounds, arms, theta, context_set, psi=None, sigma=0.01, random_state=1):
        self.context_set = context_set
        if psi is None:
            self.psi = lambda a, x: np.multiply(a, x)
        else:
            self.psi = psi
        super().__init__(n_rounds, arms, theta, sigma, random_state)

    def round(self, arm_i):
        obs_reward = (self.theta @ self.psi(self.arms[arm_i], self.context_set[self.last_context_i])
                      + self.noise[self.t])
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def get_context(self):
        self.last_context_i = self.context_indexes[self.t]
        return self.last_context_i

    def reset(self, i=0):
        super().reset(i)
        np.random.seed(self.random_state + i)
        self.context_indexes = np.random.randint(
            0, self.context_set.shape[0], self.n_rounds)
        return self


class ProductEnvironment(ContextualLinearEnvironment):
    """exp_reward = theta * psi(arm, context) + theta_p[context] * psi(arm, context)"""

    def __init__(self, n_rounds, arms, theta, context_set, theta_p, psi=None, sigma=0.01, random_state=1):
        self.theta_p = theta_p
        super().__init__(n_rounds, arms, theta, context_set, psi, sigma, random_state)

    def round(self, arm_i):
        psi = self.psi(self.arms[arm_i], self.context_set[self.last_context_i])
        obs_reward = (self.theta @ psi
                      + self.theta_p[self.last_context_i] @ psi
                      + self.noise[self.t])
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1
