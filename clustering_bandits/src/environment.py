import numpy as np


class ContextualLinearEnvironment:
    """exp_reward = theta * psi(arm, context)"""

    def __init__(self, n_rounds, arms, context_set, theta, psi, sigma=0.01, random_state=1):
        # assert context_set.shape[1] == arms.shape[1]
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
        self.reset(random_state)

    def round_all(self, pulled_arms_i):
        """computes reward for each context, action pair
        pulled_arms_i: row i contains arm j pulled for context i
        """
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


class PartitionedEnvironment(ContextualLinearEnvironment):
    """exp_reward = theta * arm[:k] + theta_p[context] * arm[k:]"""

    def __init__(self, n_rounds, arms, context_set, theta, theta_p, psi, k, sigma=0.01, random_state=1):
        self.theta_p = theta_p
        self.k = k  # first k global components
        super().__init__(n_rounds, arms, context_set, theta, psi, sigma, random_state)

    def round(self, arm_i, x_i):
        arm = self.arms[arm_i]
        obs_reward = (self.theta @ arm[:self.k]
                      + self.theta_p[x_i] @ arm[self.k:]
                      + self.noise[self.t])
        return obs_reward
