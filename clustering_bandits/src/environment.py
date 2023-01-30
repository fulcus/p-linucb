import numpy as np


class LinearEnvironment:
    """exp_reward = theta * arm"""

    def __init__(self, n_rounds, arms, theta, noise_std=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.noise_std = noise_std
        self.random_state = random_state
        self.t = None
        self.noise = None
        self.rewards = None
        self.reset(0)

    def round(self, arm_i):
        obs_reward = self.theta @ self.arms[arm_i] + self.noise[self.t]
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def reset(self, i=0):
        self.t = 0
        self.rewards = None
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.noise_std, self.n_rounds)
        return self


class ContextualLinearEnvironment(LinearEnvironment):
    """exp_reward = theta * psi(arm, context)"""

    def __init__(self, n_rounds, context_set, arms, theta, psi=None, noise_std=0.01, random_state=1):
        # super().__init__(n_rounds, arms, theta, noise_std, random_state)
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.noise_std = noise_std
        self.random_state = random_state
        self.t = None
        self.noise = None
        self.rewards = None

        self.context_set = context_set
        if psi is None:
            self.psi = lambda a, x: np.multiply(a, x)
        else:
            self.psi = psi
        self.reset()

    def round(self, arm_i):
        obs_reward = (self.theta @ self.psi(self.arms[arm_i], self.context_set[self.last_context_i])
                      + self.noise[self.t])
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1

    def get_context(self):
        self.last_context_i = self.context_indexes[self.t]
        return self.last_context_i

    def reset(self, i=0):
        self.t = 0
        self.rewards = None
        np.random.seed(self.random_state + i)
        self.context_indexes = np.random.randint(
            0, self.context_set.shape[0], self.n_rounds)
        self.noise = np.random.normal(0, self.noise_std, self.n_rounds)
        # self.contexts = np.array([self.context_set[i] for i in indexes])
        return self


class ProductEnvironment(ContextualLinearEnvironment):
    """exp_reward = theta * psi(arm, context) + theta_p[context] * psi(arm, context)"""

    def __init__(self, n_rounds, arms, contexts, theta, theta_p, psi=None, noise_std=0.01, random_state=1):
        super().__init__(n_rounds, arms, contexts, theta, psi, noise_std, random_state)
        self.theta_p = theta_p
        self.reset()

    def round(self, arm_i):
        context = self.context_set[self.last_context_i]
        psi = self.psi(self.arms[arm_i], context)
        obs_reward = (self.theta @ psi
                      + self.theta_p[self.last_context_i] @ psi
                      + self.noise[self.t])
        self.rewards = np.append(self.rewards, obs_reward)
        self.t += 1
