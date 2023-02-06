import numpy as np


class ContextualLinearEnvironment:
    """exp_reward = theta * psi(arm, context)"""

    def __init__(self, n_rounds, arms, theta, context_set, psi, sigma=0.01, random_state=1):
        self.n_rounds = n_rounds
        self.arms = arms
        self.theta = theta
        self.context_set = context_set
        self.psi = psi
        self.sigma = sigma
        self.random_state = random_state
        self.t = None
        self.noise = None
        self.rewards = np.array([])
        self.n_contexts = self.context_set.shape[0]
        self.contexts_i = np.arange(self.n_contexts)
        self.reset(0)

    def round_all(self, pulled_arms_i):
        """computes reward for each context, action pair
        pulled_arms_i: row i contains arm j pulled for context i
        """
        assert self.context_set.shape[1] == self.arms.shape[1]

        obs_reward = np.zeros((self.n_contexts,))
        for x_i, a_j in enumerate(pulled_arms_i):
            obs_reward[x_i] = self.round(a_j, x_i)
        # logging sum of rewards
        self.rewards = np.append(self.rewards, obs_reward.sum())
        self.t += 1
        return obs_reward

    def round(self, arm_i, x_i):
        psi = self.psi(self.arms[arm_i], self.context_set[x_i])
        obs_reward = (self.theta @ psi
                      + self.noise[self.t])
        return obs_reward

    def get_contexts(self):
        return self.contexts_i

    def reset(self, i=0):
        self.t = 0
        self.rewards = np.array([])
        np.random.seed(self.random_state + i)
        self.noise = np.random.normal(0, self.sigma, self.n_rounds)
        return self


class ProductEnvironment(ContextualLinearEnvironment):
    """exp_reward = theta * psi(arm, context) + theta_p[context] * psi(arm, context)"""

    def __init__(self, n_rounds, arms, theta, context_set, theta_p, psi, sigma=0.01, random_state=1):
        self.theta_p = theta_p
        super().__init__(n_rounds, arms, theta, context_set, psi, sigma, random_state)

    def round(self, arm_i, x_i):
        psi = self.psi(self.arms[arm_i], self.context_set[x_i])
        obs_reward = (self.theta @ psi
                      + self.theta_p[x_i] @ psi
                      + self.noise[self.t])
        return obs_reward
