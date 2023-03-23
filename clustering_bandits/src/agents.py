from abc import ABC, abstractmethod
import numpy as np
from src.utils import moving_mape


class Agent(ABC):
    def __init__(self, arms, n_contexts):
        self.arms = arms
        self.n_contexts = n_contexts
        # self.n_arms = arms.shape[0]
        # self.arm_dim = arms.shape[1]
        self.t = 0
        self.a_hist = []
        self.last_pulls = []

    @abstractmethod
    def pull_arm(self, context_i):
        pass

    @abstractmethod
    def update(self, reward, *args, **kwargs):
        self.t += 1


class Clairvoyant(Agent):
    """Always pulls the optimal arm"""

    def __init__(self, arms, n_contexts, theta, theta_p, k):
        super().__init__(arms, n_contexts)
        self.theta = theta
        self.theta_p = theta_p
        self.k = k

    def pull_arm(self, context_i):
        self.last_c_i = context_i
        exp_rewards = []
        for arm in self.arms[context_i]:
            exp_rewards.append(self.theta @ arm[:self.k]
                               + self.theta_p[context_i] @ arm[self.k:])
        arm_i = np.argmax(exp_rewards)
        self.a_hist.append(arm_i)
        return self.arms[context_i][arm_i]

    def update(self, reward, arm):
        self.t += 1

#  TODO refactor or delete


class UCB1Agent(Agent):
    def __init__(self, arms, n_contexts, max_reward=1):
        super().__init__(arms, n_contexts)
        self.max_reward = max_reward
        self.avg_reward = np.ones(self.n_arms) * np.inf
        self.n_pulls = np.zeros(self.n_arms)
        self.arm_index = {tuple(arm): i for i, arm in enumerate(arms)}

    def pull_arm(self, *args, **kwargs):
        ucb1 = [self.avg_reward[a] + self.max_reward *
                np.sqrt(2 * np.log(self.t)
                / self.n_pulls[a]) for a in range(self.n_arms)]
        arm_i = np.argmax(ucb1)
        self.n_pulls[arm_i] += 1
        self.a_hist.append(arm_i)
        return self.arms[arm_i]

    def update(self, reward, arm, *args, **kwargs):
        arm_i = self.arm_index[tuple(arm)]

        if self.n_pulls[arm_i] == 1:
            self.avg_reward[arm_i] = reward
        else:
            self.avg_reward[arm_i] = ((
                self.avg_reward[arm_i]
                * self.n_pulls[arm_i] + reward)
                / (self.n_pulls[arm_i] + 1))
        self.t += 1


class LinUCBAgent(Agent):
    def __init__(self, arms, n_contexts, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma=1):
        super().__init__(arms, n_contexts)
        self.n_arms = arms.shape[0]
        self.arm_dim = arms.shape[1]

        assert lmbd > 0
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma

        self.V_t = self.lmbd * np.eye(self.arm_dim)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.b_vect = np.zeros((self.arm_dim, 1))
        self.theta_hat = np.zeros((self.arm_dim, 1))

        self.last_ucb = np.zeros(self.n_arms)
        self.reward_hist = []
        self.pred_reward_hist = []
        self.arm_index = {tuple(arm): i for i, arm in enumerate(arms)}

    def update_hist(self, arm_whole, reward, pred_reward=None):
        arm_i = self.arm_index[tuple(arm_whole)]
        self.a_hist.append(arm_i)
        self.reward_hist.append(reward)
        if pred_reward is not None:
            self.pred_reward_hist.append(pred_reward)

    def pull_arm(self, *args, **kwargs):
        if len(self.a_hist) < self.arm_dim:
            arm_i = len(self.a_hist) % self.n_arms
        else:
            arm_i = self._estimate_linucb_arm()
        # self.a_hist.append(arm_i)
        return self.arms[arm_i]

    def update(self, reward, arm, *args, **kwargs):
        arm = arm.reshape(-1, 1)
        # update params
        self.V_t += arm @ arm.T
        self.b_vect = self.b_vect + arm * reward
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.theta_hat = self.V_t_inv @ self.b_vect
        # update hist
        # self.reward_hist.append(reward)
        self.t += 1

    def _estimate_linucb_arm(self):
        bound = self._beta_t_fun_linucb()
        for i, arm in enumerate(self.arms):
            arm = arm.reshape(-1, 1)
            self.last_ucb[i] = (self.theta_hat.T @ arm + bound
                                * np.sqrt(arm.T @ self.V_t_inv @ arm))
        return np.argmax(self.last_ucb)

    def _beta_t_fun_linucb(self):
        return self.max_theta_norm * np.sqrt(self.lmbd) + \
            np.sqrt(2 * self.sigma**2 * np.log(self.t) +
                    np.log(np.linalg.det(self.V_t) / (self.lmbd ** self.arm_dim)))


class INDLinUCBAgent(Agent):
    """One independent LinUCBAgent instance per context - BASELINE"""

    def __init__(self, arms, n_contexts, horizon, lmbd,
                 max_theta_norm, max_arm_norm, sigma=1):
        super().__init__(arms, n_contexts)
        self.lmbd = lmbd
        self.horizon = horizon
        self.max_theta_norm = max_theta_norm
        self.max_arm_norm = max_arm_norm
        self.sigma = sigma
        # self.arm_index = {tuple(arm): i for i, arm in enumerate(arms)}
        self.context_agent = [
            LinUCBAgent(
                self.arms[i],
                self.n_contexts,
                self.horizon,
                self.lmbd,
                self.max_theta_norm,
                self.max_arm_norm,
                self.sigma)
            for i in range(self.n_contexts)
        ]

    def pull_arm(self, context_i):
        self.last_c_i = context_i
        arm = self.context_agent[context_i].pull_arm()
        return arm

    def update(self, reward, arm):
        self.context_agent[self.last_c_i].update_hist(arm, reward)
        self.context_agent[self.last_c_i].update(
            reward, arm)
        self.t += 1


class PartitionedAgentStatic(INDLinUCBAgent):
    """An independent linear bandit per context.
    The first bandit that learns a good approximation of theta fixes
    the first k components of theta for all."""

    def __init__(self, arms, n_contexts, horizon, lmbd,
                 max_theta_norm, max_theta_norm_local, max_arm_norm, max_arm_norm_local, k=2, err_th=0.1, win=10, sigma=1):
        super().__init__(arms, n_contexts, horizon, lmbd,
                         max_theta_norm, max_arm_norm, sigma)
        self.max_theta_norm_local = max_theta_norm_local
        self.max_arm_norm_local = max_arm_norm_local
        self.k = k
        self.err_th = err_th
        self.win = win

        self.is_split = False
        self.t_split = None
        self.reward_global = None
        self.subarm_global = None
        self.subtheta_global = None
        # might use later to log errors or delta theta
        self.err_hist = [[] for _ in range(self.n_contexts)]

    def pull_arm(self, context_i):
        self.last_c_i = context_i
        arm = self.context_agent[context_i].pull_arm()
        # if split has happened arm_i is index of second half
        if self.is_split:
            arm = np.concatenate([self.subarm_global, arm])
        return arm

    def update(self, reward, arm):
        agent = self.context_agent[self.last_c_i]
        if not self.is_split:
            pred_reward = agent.theta_hat.T @ arm
            agent.update_hist(arm, reward, pred_reward.squeeze())
            agent.update(reward, arm)
            if moving_mape(agent.reward_hist, agent.pred_reward_hist, win=self.win) <= self.err_th:
                self.subtheta_global = agent.theta_hat[:self.k]
                self.subarm_global = arm[:self.k]
                self.reward_global = self.subtheta_global.T @ self.subarm_global

                self.is_split = True
                self.t_split = self.t

                self._split_agents_params()
                self._print_debug(agent, arm)
        else:
            local_arm = arm[self.k:]
            # remove global arm contribution to reward for local arm update
            pred_reward = self.reward_global + agent.theta_hat.T @ local_arm
            agent.update_hist(arm, reward, pred_reward.squeeze())
            reward -= self.reward_global
            agent.update(reward, local_arm)
        self.t += 1

    def _split_agents_params(self):
        for c_i, agent in enumerate(self.context_agent):
            dim_local = agent.arm_dim - self.k
            arms_global = np.delete(agent.arms, np.s_[self.k:], axis=1)
            arms_local = np.delete(agent.arms, np.s_[:self.k], axis=1)
            # remove global components from all agents
            agent.arm_dim = dim_local
            agent.max_arm_norm = self.max_arm_norm_local
            agent.max_theta_norm = self.max_theta_norm_local
            agent.arms = arms_local
            # removing global components contributions
            y_loc = np.array(agent.reward_hist) - \
                self.subtheta_global.T @ arms_global[agent.a_hist].T
            A_loc = arms_local[agent.a_hist]
            # recompute bandit parameters
            agent.V_t = A_loc.T @ A_loc + agent.lmbd * np.eye(agent.arm_dim)
            agent.b_vect = A_loc.T @ y_loc.T
            agent.V_t_inv = np.linalg.inv(agent.V_t)
            agent.theta_hat = agent.V_t_inv @ agent.b_vect

    def _print_debug(self, agent, arm):
        print(f"ma={moving_mape(agent.reward_hist, agent.pred_reward_hist, win=self.win)}\n" +
              f"theta_hat={agent.theta_hat.squeeze()}\n" +
              f"arm_leader={arm.squeeze()}\n" +
              f"{self.last_c_i=}\n" +
              f"t_split={self.t_split}")
