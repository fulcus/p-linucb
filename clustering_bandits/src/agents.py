from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self, arms, random_state=1):
        self.arms = arms
        self.random_state = random_state
        self.n_arms = arms.shape[0]
        self.arm_dim = arms.shape[1]
        self.reset()

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, reward, *args, **kwargs):
        pass

    def reset(self):
        self.t = 0
        self.a_hist = []
        self.last_pull_i = None


class Clairvoyant(Agent):
    """Always pulls the optimal arm"""

    def __init__(self, arms, theta, psi, theta_p, context_set, random_state=1):
        self.context_set = context_set
        self.theta = theta
        self.theta_p = theta_p
        self.psi = psi
        super().__init__(arms, random_state)

    def pull_arm(self, context_i):
        exp_rewards = np.zeros(self.n_arms)
        context = self.context_set[context_i]
        for i, arm in enumerate(self.arms):
            psi = self.psi(arm, context)
            exp_rewards[i] = (self.theta @ psi
                              + self.theta_p[context_i] @ psi)
        self.last_pull_i = np.argmax(exp_rewards)
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward):
        self.t += 1


class UCB1Agent(Agent):
    def __init__(self, arms, max_reward=1, random_state=1):
        self.max_reward = max_reward
        super().__init__(arms, random_state)

    def reset(self):
        super().reset()
        self.avg_reward = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self, context_i=None):
        ucb1 = [self.avg_reward[a] + self.max_reward *
                np.sqrt(2 * np.log(self.t)
                / self.n_pulls[a]) for a in range(self.n_arms)]
        self.last_pull_i = np.argmax(ucb1)
        self.n_pulls[self.last_pull_i] += 1
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward):
        self.avg_reward[self.last_pull_i] = ((
            self.avg_reward[self.last_pull_i]
            * self.n_pulls[self.last_pull_i] + reward)
            / (self.n_pulls[self.last_pull_i] + 1))
        self.t += 1


class LinUCBAgent(Agent):
    def __init__(self, arms, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma=1, random_state=1):
        assert lmbd > 0
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma
        super().__init__(arms, random_state)

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.arm_dim)
        self.b_vect = np.zeros((self.arm_dim, 1))
        self.theta_hat = np.zeros((self.arm_dim, 1))
        self.last_ucb = np.zeros(self.n_arms)
        self.first = True
        return self

    def pull_arm(self, context_i=None):
        if self.first:
            np.random.seed(self.random_state)
            self.last_pull_i = int(np.random.uniform(high=self.n_arms))
            self.first = False
        else:
            _, self.last_pull_i = self._estimate_linucb_arm()
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward):
        last_pull = self.arms[self.last_pull_i].reshape(self.arm_dim, 1)
        self.V_t = self.V_t + (last_pull @ last_pull.T)
        self.b_vect = self.b_vect + last_pull * reward
        self.theta_hat = np.linalg.inv(self.V_t) @ self.b_vect
        self.t += 1

    def _estimate_linucb_arm(self):
        bound = self._beta_t_fun_linucb()
        for i, arm in enumerate(self.arms):
            arm = arm.reshape(self.arm_dim, 1)
            self.last_ucb[i] = (self.theta_hat.T @ arm + bound
                                * np.sqrt(arm.T @ np.linalg.inv(self.V_t) @ arm))
        return self.arms[np.argmax(self.last_ucb), :], np.argmax(self.last_ucb)

    def _beta_t_fun_linucb(self):
        return (self.max_theta_norm * np.sqrt(self.lmbd)
                + np.sqrt(2 * self.sigma ** 2 * np.log(self.horizon)
                + (self.arm_dim * np.log(
                    (self.arm_dim * self.lmbd
                     + self.horizon * (self.max_arm_norm ** 2))
                    / (self.arm_dim * self.lmbd)
                ))))


class ContextualLinUCBAgent(LinUCBAgent):
    def __init__(self, arms, context_set, psi, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma=1, random_state=1):
        self.context_set = context_set
        self.psi = psi
        super().__init__(arms, horizon, lmbd,
                         max_theta_norm, max_arm_norm, sigma, random_state)

    def pull_arm(self, context_i):
        self.last_context = self.context_set[context_i]
        if self.first:
            np.random.seed(self.random_state)
            self.last_pull_i = int(np.random.uniform(high=self.n_arms))
            self.first = False
        else:
            _, self.last_pull_i = self._estimate_linucb_arm()
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward):
        last_psi = self.psi(self.arms[self.last_pull_i], self.last_context)
        last_psi = last_psi.reshape(self.arm_dim, 1)
        self.V_t = self.V_t + (last_psi @ last_psi.T)
        self.b_vect = self.b_vect + last_psi * reward
        self.theta_hat = np.linalg.inv(self.V_t) @ self.b_vect
        self.t += 1

    def _estimate_linucb_arm(self):
        bound = self._beta_t_fun_linucb()
        for i, arm in enumerate(self.arms):
            psi = self.psi(arm, self.last_context)
            psi = psi.reshape(self.arm_dim, 1)
            self.last_ucb[i] = (self.theta_hat.T @ psi + bound
                                * np.sqrt(psi.T @ np.linalg.inv(self.V_t) @ psi))
        return self.arms[np.argmax(self.last_ucb), :], np.argmax(self.last_ucb)


class INDUCB1Agent(Agent):
    """One independent UCB1Agent instance per context"""

    def __init__(self, arms, context_set, max_reward=1, random_state=1):
        self.context_set = context_set
        self.max_reward = max_reward
        super().__init__(arms, random_state)

    def reset(self):
        super().reset()
        self.context_agent = [
            UCB1Agent(
                self.arms,
                self.max_reward,
                self.random_state)
            for _ in range(len(self.context_set))
        ]

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm + theta_p * arm)"""
        self.last_context_i = context_i
        arm_i = self.context_agent[context_i].pull_arm()
        self.a_hist.append(arm_i)
        return arm_i

    def update(self, reward):
        self.context_agent[self.last_context_i].update(reward)


class INDLinUCBAgent(Agent):
    """One independent LinUCBAgent instance per context"""

    def __init__(self, arms, context_set, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma=1, random_state=1):
        self.context_set = context_set
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma
        super().__init__(arms, random_state)

    def reset(self):
        super().reset()
        self.context_agent = [
            LinUCBAgent(
                self.arms,
                self.horizon,
                self.lmbd,
                self.max_theta_norm,
                self.max_arm_norm,
                self.sigma,
                self.random_state)
            for _ in range(len(self.context_set))
        ]

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm + theta_p * arm)"""
        self.last_context_i = context_i
        arm_i = self.context_agent[context_i].pull_arm()
        self.a_hist.append(arm_i)
        return arm_i

    def update(self, reward):
        self.context_agent[self.last_context_i].update(reward)


class ProductContextualAgent(Agent):
    """Combines a global (contextual) linear bandit 
    and an independent linear bandit per context"""

    def __init__(self, arms, context_set, psi, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma, random_state=1):
        self.context_set = context_set
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma
        self.psi = psi
        super().__init__(arms, random_state)

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm + theta_p * arm)"""
        self.last_context_i = context_i

        # compute and combine their ucb
        self.agent_global.pull_arm(context_i)
        self.context_agent[context_i].pull_arm()
        ucb = (self.agent_global.last_ucb
               + self.context_agent[context_i].last_ucb)
        self.last_pull_i = np.argmax(ucb)

        self.agent_global.last_pull_i = self.last_pull_i
        self.context_agent[context_i].last_pull_i = self.last_pull_i
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward):
        self.agent_global.update(reward)
        psi = self.psi(self.arms[self.agent_global.last_pull_i],
                       self.context_set[self.last_context_i])
        pred_reward = self.agent_global.theta_hat.T @ psi
        residual = reward - pred_reward
        self.context_agent[self.last_context_i].update(residual)

    def reset(self):
        super().reset()
        self.agent_global = ContextualLinUCBAgent(
            self.arms, self.context_set, self.psi, self.horizon, self.lmbd,
            self.max_theta_norm, self.max_arm_norm, self.random_state)
        self.context_agent = [
            LinUCBAgent(
                self.arms, self.horizon,
                self.lmbd, self.max_theta_norm,
                self.max_arm_norm, self.random_state)
            for _ in range(len(self.context_set))
        ]


class ProductLinearAgent(ProductContextualAgent):
    """Combines a global linear bandit 
    and an independent linear bandit per context"""

    def __init__(self, arms, context_set, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma, random_state=1):
        def psi(a, x): return a
        super().__init__(arms, context_set, psi, horizon, lmbd,
                         max_theta_norm, max_arm_norm, sigma, random_state)
