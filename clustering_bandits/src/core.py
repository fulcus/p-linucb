import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from src.environment import ContextualLinearEnvironment


class Core:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def simulation(self, n_epochs, n_rounds, parallel=False):
        args = [(deepcopy(self.agent), deepcopy(
            self.environment.reset(i)), n_rounds) for i in range(n_epochs)]
        rewards = []
        a_hists = []
        if parallel:
            with ProcessPoolExecutor(max_workers=4) as executor:
                for result in executor.map(self.helper, args):
                    rewards.append(result[0])
                    a_hists.append(result[1])
        else:
            for arg in args:
                result = self.helper(arg)
                # result[0] list of rewards
                rewards.append(result[0])
                a_hists.append(result[1])
        return np.array(rewards), np.array(a_hists)

    def helper(self, arg):
        return self.epoch(arg[0], arg[1], arg[2])

    def epoch(self, agent, environment, n_rounds=10):
        for _ in range(n_rounds):
            if isinstance(environment, ContextualLinearEnvironment):
                x_i = environment.get_context()
                new_a = agent.pull_arm(x_i)
            else:
                new_a = agent.pull_arm()
            environment.round(new_a)
            agent.update(
                environment.rewards[-1])
        return (environment.rewards, agent.a_hist)
