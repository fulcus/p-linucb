import numpy as np
import networkx as nx

from src.agents import Agent


class Cluster:
    def __init__(self, users, M, b, N):
        self.users = users  # a list/array of users
        self.M = M
        self.b = b
        self.N = N
        self.M_inv = np.linalg.inv(self.M)
        self.theta = np.matmul(self.M_inv, self.b)

    def pull_arm(self, arms, t):
        return np.argmax(np.dot(arms, self.theta) + self._beta(arms.shape[1], t) * (np.matmul(arms, self.M_inv) * arms).sum(axis=1))

    def _beta(self, d, t):
        return np.sqrt(d * np.log(1 + self.N / d) + 4 * np.log(t) + np.log(2)) + 1

    def update(self, a, y):
        self.M += np.outer(a, a)
        self.b += y * a
        self.N += 1

        self.M_inv = np.linalg.inv(self.M)
        self.theta = np.matmul(self.M_inv, self.b)


class CLUB(Agent):
    # random_init: use random initialization or not
    def __init__(self, arms, context_set, horizon, edge_probability=1):
        super().__init__(arms, context_set)
        self.context_indexes = range(self.n_contexts)
        # Base
        self.horizon = horizon

        # INDLinUCB
        # one LinUCB per user i
        self.M = {i: np.eye(self.arm_dim) for i in range(self.n_contexts)}
        self.b = {i: np.zeros(self.arm_dim) for i in range(self.n_contexts)}
        self.M_inv = {i: np.eye(self.arm_dim) for i in range(self.n_contexts)}
        self.theta = {i: np.zeros(self.arm_dim)
                      for i in range(self.n_contexts)}
        self.num_pulls = np.zeros(self.n_contexts)  # users array

        # CLUB
        self.nu = self.n_contexts
        # self.alpha = 4 * np.sqrt(d) # parameter for cut edge
        self.G = nx.gnp_random_graph(self.n_contexts, edge_probability)
        self.clusters = {0: Cluster(users=range(
            self.n_contexts), M=np.eye(self.arm_dim), b=np.zeros(self.arm_dim), N=0)}
        # index represents user, value is index of cluster he belongs to
        self.cluster_inds = np.zeros(self.n_contexts)
        # num_clusters over time (increasing)
        self.num_clusters = np.zeros(horizon)

    def pull_arm(self, context_i):
        # get cluster of user i
        cluster_i = self.cluster_inds[context_i]
        cluster = self.clusters[cluster_i]
        self.last_pull_i = cluster.pull_arm(self.arms, self.t)
        self.a_hist.append(self.last_pull_i)
        return self.last_pull_i

    def update_arms(self, rewards):
        for arm_i, reward, c_i in zip(self.last_pulls_i, rewards, range(self.n_contexts)):
            # update weights
            self.update(i=c_i, a=self.arms[arm_i], y=reward)
        self.update_clustering(self.t)
        self.t += 1

    def update(self, i, a, y):
        # INDLinUCB
        self.M[i] += np.outer(a, a)
        self.b[i] += y * a
        self.num_pulls[i] += 1
        self.M_inv[i] = np.linalg.inv(self.M[i])
        self.theta[i] = np.matmul(self.M_inv[i], self.b[i])
        # CLUB
        cluster_i = self.cluster_inds[i]
        cluster = self.clusters[cluster_i]
        cluster.update(a, y)

    def update_clustering(self, t):
        update_clusters = False
        # delete edges
        for i in self.context_indexes:
            cluster_i = self.cluster_inds[i]
            A = [a for a in self.G.neighbors(i)]
            for j in A:
                if self.num_pulls[i] and self.num_pulls[j] and self._if_split(self.theta[i] - self.theta[j], self.num_pulls[i], self.num_pulls[j]):
                    self.G.remove_edge(i, j)
                    # print(f"remove_edge({i},{j})")
                    update_clusters = True

        if update_clusters:
            C = set()  # contexts
            for i in self.context_indexes:  # suppose there is only one user per round
                C = nx.node_connected_component(self.G, i)
                cluster_i = self.cluster_inds[i]
                if len(C) < len(self.clusters[cluster_i].users):
                    remain_users = set(self.clusters[cluster_i].users)
                    self.clusters[cluster_i] = Cluster(list(C), M=sum([self.M[k]-np.eye(self.arm_dim) for k in C])+np.eye(
                        self.arm_dim), b=sum([self.b[k] for k in C]), N=sum([self.num_pulls[k] for k in C]))

                    remain_users = remain_users - set(C)
                    cluster_i = max(self.clusters) + 1
                    while len(remain_users) > 0:
                        j = np.random.choice(list(remain_users))
                        C = nx.node_connected_component(self.G, j)

                        self.clusters[cluster_i] = Cluster(list(C), M=sum([self.M[k]-np.eye(self.arm_dim) for k in C])+np.eye(
                            self.arm_dim), b=sum([self.b[k] for k in C]), N=sum([self.num_pulls[k] for k in C]))
                        for j in C:
                            self.cluster_inds[j] = cluster_i

                        cluster_i += 1
                        remain_users = remain_users - set(C)
            print(len(self.clusters))
        self.num_clusters[t] = len(self.clusters)

    def _if_split(self, theta, N1, N2):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1

        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))
        return np.linalg.norm(theta) > alpha * (_factT(N1) + _factT(N2))
