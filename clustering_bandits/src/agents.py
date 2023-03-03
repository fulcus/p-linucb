from abc import ABC, abstractmethod
import numpy as np
from src.utils import psi_lin


class Agent(ABC):
    def __init__(self, arms, context_set):
        #  assert context_set.shape[1] == arms.shape[1]
        self.arms = arms
        self.context_set = context_set
        self.n_contexts = context_set.shape[0]
        self.n_arms = arms.shape[0]
        self.arm_dim = arms.shape[1]
        self.t = 0
        self.a_hist = []
        self.last_pull_i = None
        self.last_pulls_i = np.zeros(self.n_contexts, dtype=int)

    @abstractmethod
    def pull_arm(self, context_i):
        pass

    def pull_arms(self, context_indexes):
        for c_i in context_indexes:
            self.last_pulls_i[c_i] = self.pull_arm(c_i)
        return self.last_pulls_i

    @abstractmethod
    def update(self, reward, *args, **kwargs):
        pass

    def update_arms(self, rewards):
        for arm_i, reward, c_i in zip(self.last_pulls_i, rewards, range(self.n_contexts)):
            self.update(reward, arm_i=arm_i, context_i=c_i)
        self.t += 1


class Clairvoyant(Agent):
    """Always pulls the optimal arm"""

    def __init__(self, arms, context_set, theta, theta_p, k, psi):
        super().__init__(arms, context_set)
        self.theta = theta
        self.theta_p = theta_p
        self.psi = psi
        self.k = k

    def pull_arm(self, context_i):
        exp_rewards = np.zeros(self.n_arms)
        context = self.context_set[context_i]
        for i, arm in enumerate(self.arms):
            #  psi = self.psi(arm, context)
            exp_rewards[i] = (self.theta @ arm[:self.k]
                              + self.theta_p[context_i] @ arm[self.k:])
        # TODO can remove last_pull_i from everywhere,
        # since it's always given as parameter to update
        self.last_pull_i = np.argmax(exp_rewards)
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward, arm_i=None, context_i=None):
        if arm_i is not None:
            self.last_pull_i = arm_i


class UCB1Agent(Agent):
    def __init__(self, arms, context_set, max_reward=1):
        super().__init__(arms, context_set)
        self.max_reward = max_reward
        self.avg_reward = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)

    def pull_arm(self, context_i=None):
        ucb1 = [self.avg_reward[a] + self.max_reward *
                np.sqrt(2 * np.log(self.t)
                / self.n_pulls[a]) for a in range(self.n_arms)]
        self.last_pull_i = np.argmax(ucb1)
        self.n_pulls[self.last_pull_i] += 1
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward, arm_i=None, context_i=None):
        if arm_i is not None:
            self.last_pull_i = arm_i
        self.avg_reward[arm_i] = ((
            self.avg_reward[arm_i]
            * self.n_pulls[arm_i] + reward)
            / (self.n_pulls[arm_i] + 1))


class LinUCBAgent(Agent):
    def __init__(self, arms, context_set, horizon, lmbd,
                 max_theta_norm_sum, max_arm_norm, sigma=1):
        super().__init__(arms, context_set)
        assert lmbd > 0
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm_sum = max_theta_norm_sum
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma

        self.last_pull_i = 0
        self.V_t = self.lmbd * np.eye(self.arm_dim)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.b_vect = np.zeros((self.arm_dim, 1))
        self.theta_hat = np.zeros((self.arm_dim, 1))

        self.last_ucb = np.zeros(self.n_arms)
        self.reward_hist = []
        self.theta_hat_hist = []
        self.mse_hist = []

    def pull_arm(self, context_i=None):
        if len(self.a_hist) < self.arm_dim:
            self.last_pull_i = len(self.a_hist) % self.n_arms
        else:
            self.last_pull_i = self._estimate_linucb_arm()
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward, arm_i=None, context_i=None):
        if arm_i is not None:
            self.last_pull_i = arm_i
        last_pull = self.arms[self.last_pull_i].reshape(-1, 1)
        # update params
        self.V_t = self.V_t + (last_pull @ last_pull.T)
        self.b_vect = self.b_vect + last_pull * reward
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.theta_hat = self.V_t_inv @ self.b_vect
        # update hist
        self.reward_hist.append(reward)
        self.theta_hat_hist.append(self.theta_hat)
        self.mse_hist.append(reward - self.theta_hat.T @ last_pull)

    def _estimate_linucb_arm(self):
        bound = self._beta_t_fun_linucb()
        for i, arm in enumerate(self.arms):
            arm = arm.reshape(-1, 1)
            self.last_ucb[i] = (self.theta_hat.T @ arm + bound
                                * np.sqrt(arm.T @ self.V_t_inv @ arm))
        return np.argmax(self.last_ucb)

    def _beta_t_fun_linucb(self):
        return (self.max_theta_norm_sum * np.sqrt(self.lmbd)
                + np.sqrt(2 * self.sigma ** 2 * np.log(self.horizon)
                + (self.arm_dim * np.log(
                    (self.arm_dim * self.lmbd
                     + self.horizon * (self.max_arm_norm ** 2))
                    / (self.arm_dim * self.lmbd)
                ))))


class ContextualLinUCBAgent(LinUCBAgent):
    def __init__(self, arms, context_set, psi, psi_dim, horizon, lmbd,
                 max_theta_norm_sum, max_arm_norm, sigma=1):
        super().__init__(arms, context_set, horizon, lmbd,
                         max_theta_norm_sum, max_arm_norm, sigma)
        self.psi = psi
        self.psi_dim = psi_dim
        self.V_t = self.lmbd * np.eye(self.psi_dim)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.b_vect = np.zeros((self.psi_dim, 1))
        self.theta_hat = np.zeros((self.psi_dim, 1))

    def pull_arm(self, context_i):
        if len(self.a_hist) < self.arm_dim:
            self.last_pull_i = len(self.a_hist) % self.n_arms
        else:
            self.last_pull_i = self._estimate_linucb_arm(context_i)
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update(self, reward, arm_i=None, context_i=None):
        if arm_i is not None:
            self.last_pull_i = arm_i

        last_psi = self.psi(
            self.arms[self.last_pull_i], self.context_set[context_i])
        last_psi = last_psi.reshape(-1, 1)
        self.V_t = self.V_t + (last_psi @ last_psi.T)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.b_vect = self.b_vect + last_psi * reward
        self.theta_hat = self.V_t_inv @ self.b_vect

    def _estimate_linucb_arm(self, context_i):
        bound = self._beta_t_fun_linucb()
        for i, arm in enumerate(self.arms):
            psi = self.psi(arm, self.context_set[context_i])
            psi = psi.reshape(-1, 1)
            # TODO fix dim
            self.last_ucb[i] = (self.theta_hat.T @ psi + bound
                                * np.sqrt(psi.T @ self.V_t_inv @ psi))
        return np.argmax(self.last_ucb)


class INDUCB1Agent(Agent):
    """One independent UCB1Agent instance per context"""

    def __init__(self, arms, context_set, max_reward=1):
        super().__init__(arms, context_set)
        self.context_agent = [
            UCB1Agent(
                self.arms,
                self.context_set,
                max_reward)
            for _ in range(self.n_contexts)
        ]

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm + theta_p * arm)"""
        arm_i = self.context_agent[context_i].pull_arm()
        self.a_hist.append(arm_i)
        return arm_i

    def update(self, reward, arm_i, context_i):
        self.context_agent[context_i].update(
            reward, arm_i, context_i)


class INDLinUCBAgent(Agent):
    """One independent LinUCBAgent instance per context"""

    def __init__(self, arms, context_set, horizon, lmbd,
                 max_theta_norm_sum, max_arm_norm, sigma=1):
        super().__init__(arms, context_set)
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm_sum = max_theta_norm_sum
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma
        self.context_agent = [
            LinUCBAgent(
                self.arms,
                self.context_set,
                self.horizon,
                self.lmbd,
                self.max_theta_norm_sum,
                self.max_arm_norm,
                self.sigma)
            for _ in range(self.n_contexts)
        ]

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm + theta_p * arm)"""
        arm_i = self.context_agent[context_i].pull_arm()
        self.a_hist.append(arm_i)
        return arm_i

    def update(self, reward, arm_i=None, context_i=None):
        if arm_i is not None:
            self.last_pull_i = arm_i
        self.context_agent[context_i].update(
            reward, self.last_pull_i, context_i)


class PartitionedAgent(INDLinUCBAgent):
    """An independent linear bandit per context. 
    The first bandit that learns a good approximation of theta fixes 
    the first k components of theta for all."""

    def __init__(self, arms, context_set, horizon, lmbd,
                 max_theta_norm_sum, max_arm_norm, k, err_th, sigma=1):
        super().__init__(arms, context_set, horizon, lmbd,
                         max_theta_norm_sum, max_arm_norm, sigma)
        self.k = k
        self.err_th = err_th
        self.subtheta_global = None
        self.subarm_global = None
        self.t_split = None
        self.arm_index = {tuple(arm): i for i, arm in enumerate(arms)}

        self.arms_global = np.delete(self.arms, np.s_[self.k:], axis=1)
        # self.arms_local = np.unique(
        #     np.delete(self.arms, np.s_[:self.k], axis=1), axis=0)
        self.arms_local = np.delete(self.arms, np.s_[:self.k], axis=1)
        self.max_arm_norm_local = np.max(
            [np.linalg.norm(a) for a in self.arms_local])

    def pull_arm(self, context_i):
        """arm = argmax (theta * arm[:k] + theta_p[context] * arm[k:])"""
        arm_i = self.context_agent[context_i].pull_arm()
        # if split has happened arm_i is index of second half
        if self.subarm_global is not None:
            subarm_local = self.arms_local[arm_i]
            arm = np.concatenate([self.subarm_global, subarm_local])
            arm_i = self.arm_index[tuple(arm)]
        self.a_hist.append(arm_i)
        return arm_i

    def update_arms(self, rewards):
        arm_leader = c_i_leader = None
        for arm_i, reward, c_i in zip(self.last_pulls_i, rewards, range(self.n_contexts)):
            if self.subtheta_global is None:
                pred_reward = self.context_agent[c_i].theta_hat.T @ self.arms[arm_i]
                # error below threshold
                if (reward - pred_reward) ** 2 <= self.err_th:
                    arm_leader, c_i_leader = self.arms[arm_i], c_i
            else:
                # remove global arm contribution to update local arm
                reward -= self.subtheta_global.T @ self.arms_global[self.last_pull_i]
            self.update(reward, arm_i=arm_i, context_i=c_i)
        # recompute params at the end of round
        if arm_leader is not None:
            self.t_split = self.t
            self._split_agents_params(arm_leader, c_i_leader)
        self.t += 1

    def _split_agents_params(self, arm, context_i):
        print(
            f"{self.t_split=}\n{self.context_agent[context_i].theta_hat=}\n{context_i=}")
        self.subtheta_global = self.context_agent[context_i].theta_hat[:self.k]
        self.subarm_global = arm[:self.k]
        for agent in self.context_agent:
            # remove global components from all agents
            agent.arm_dim -= self.k
            agent.max_arm_norm = self.max_arm_norm_local
            agent.arms = self.arms_local
            # removing global components contributions
            y_loc = np.array(agent.reward_hist) - \
                self.subtheta_global.T @ self.arms_global[agent.a_hist].T
            A_loc = self.arms_local[agent.a_hist]
            # recompute bandit parameters
            agent.V_t = A_loc.T @ A_loc + \
                agent.lmbd * np.eye(agent.arm_dim)
            agent.b_vect = A_loc.T @ y_loc.T
            agent.V_t_inv = np.linalg.inv(agent.V_t)
            agent.theta_hat = agent.V_t_inv @ agent.b_vect
