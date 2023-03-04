import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from src.agents import PartitionedAgent


class Core:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def simulation(self, n_epochs, n_rounds, parallel=False):
        epoch_objs = [(deepcopy(self.agent), deepcopy(
            self.environment.reset(i)), n_rounds) for i in range(n_epochs)]
        rewards = []
        a_hists = []
        t_splits = []
        err_hists = []
        if parallel:
            with ProcessPoolExecutor(max_workers=4) as executor:
                for rews_epoch, actions_epoch in executor.map(self.helper, epoch_objs):
                    rewards.append(rews_epoch)
                    a_hists.append(actions_epoch)
        else:
            for args in epoch_objs:
                if isinstance(args[0], PartitionedAgent):
                    rews_epoch, actions_epoch, t_split, err_hist = self.helper(
                        args)
                    t_splits.append(t_split)
                    err_hists.append(err_hist)
                else:
                    rews_epoch, actions_epoch = self.helper(args)
                rewards.append(rews_epoch)
                a_hists.append(actions_epoch)
        return np.array(rewards), np.array(a_hists), np.array(t_splits), np.array(err_hists)

    def helper(self, args):
        return self.epoch(args[0], args[1], args[2])

    def epoch(self, agent, environment, n_rounds=10):
        for _ in range(n_rounds):
            context_indexes = environment.get_contexts()
            actions = agent.pull_arms(context_indexes)
            # actions: one row per context
            rewards = environment.round_all(actions)
            agent.update_arms(rewards)
        if isinstance(agent, PartitionedAgent):
            return environment.rewards, agent.a_hist, agent.t_split, agent.agents_err_hist
        return environment.rewards, agent.a_hist
