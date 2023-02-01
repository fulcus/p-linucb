from src.agents import *
from src.environment import ContextualLinearEnvironment, ProductEnvironment
from src.core import Core

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import argparse
import warnings
import json
import os


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim_id', type=int, default=0)
    parser.add_argument('-t', '--test_file', default=None)
    args = parser.parse_args()

    in_dir = 'clustering_bandits/test/input/'
    out_dir = f'clustering_bandits/test/output/simulation_{args.sim_id}/'
    # os.makedirs(out_dir + 'tex/', exist_ok=True)
    os.makedirs(out_dir + 'png/', exist_ok=True)
    final_logs = {}
    test_files = os.listdir(
        in_dir) if args.test_file is None else [args.test_file]

    for testcase in test_files:
        print(f'################## Testcase {testcase} ###################')

        with open(f'{in_dir}{testcase}') as f:
            param_dict = json.load(f)
        testcase, _ = testcase.split('.')
        # list to np.ndarray
        for k, v in param_dict.items():
            if type(v) == list:
                param_dict[k] = np.squeeze(np.asarray(v))

        agents_list = [
            UCB1Agent(param_dict["arms"],
                      max_reward,
                      param_dict['seed']),
            LinUCBAgent(param_dict["arms"],
                        param_dict["horizon"],
                        1,
                        param_dict["max_theta_norm"],
                        param_dict["max_arm_norm"],
                        param_dict['sigma'],
                        param_dict['seed']),
            ContextualLinUCBAgent(param_dict["arms"],
                                  param_dict["context_set"],
                                  None,
                                  param_dict["horizon"],
                                  1,
                                  param_dict["max_theta_norm"],
                                  param_dict["max_arm_norm"],
                                  param_dict['sigma'],
                                  param_dict['seed']),
            INDLinUCBAgent(param_dict["arms"],
                           param_dict["context_set"],
                           None,
                           param_dict["horizon"],
                           1,
                           param_dict["max_theta_norm"],
                           param_dict["max_arm_norm"],
                           param_dict['sigma'],
                           param_dict['seed']),
            ProductLinUCBAgent(param_dict["arms"],
                               param_dict["context_set"],
                               None,
                               param_dict["horizon"],
                               1,
                               param_dict["max_theta_norm"],
                               param_dict["max_arm_norm"],
                               param_dict['sigma'],
                               param_dict['seed'])
        ]

        logs = {}
        a_hists = {}
        if "theta_p" in param_dict:
            print("Product Setting")
            theta_p = param_dict["theta_p"]
            env = ProductEnvironment(n_rounds=param_dict["horizon"],
                                     arms=param_dict["arms"],
                                     context_set=param_dict["context_set"],
                                     theta=param_dict["theta"],
                                     theta_p=theta_p,
                                     sigma=param_dict['sigma'],
                                     random_state=param_dict['seed'])
        else:
            print("Contextual Setting")
            theta_p = None
            env = ContextualLinearEnvironment(n_rounds=param_dict["horizon"],
                                              arms=param_dict["arms"],
                                              theta=param_dict["theta"],
                                              context_set=param_dict["context_set"],
                                              sigma=param_dict['sigma'],
                                              random_state=param_dict['seed'])

        # Clairvoyant
        print('Training Clairvoyant Algorithm')
        agent = Clairvoyant(arms=param_dict["arms"],
                            theta=param_dict["theta"],
                            theta_p=theta_p,
                            context_set=param_dict["context_set"])
        env.reset()
        core = Core(env, agent)
        # rewards, arms
        clairvoyant_logs, a_hists['Clairvoyant'] = core.simulation(
            n_epochs=param_dict['n_epochs'], n_rounds=param_dict["horizon"])

        # Reward upper bound
        max_reward = clairvoyant_logs.max()

        # Train all agents
        for agent in agents_list:
            agent_name = agent.__class__.__name__
            print(f"Training {agent_name}")
            core = Core(env, agent)
            logs[agent_name], a_hists[agent_name] = core.simulation(
                n_epochs=param_dict["n_epochs"], n_rounds=param_dict["horizon"])

        # Regrets computing
        print('Computing regrets...')
        clairvoyant_logs = clairvoyant_logs.astype(np.float64)

        regret = {label: np.inf *
                  np.ones((param_dict['n_epochs'], param_dict["horizon"])) for label in logs.keys()}
        for label in regret.keys():
            logs[label] = logs[label].astype(np.float64)
            for i in range(param_dict['n_epochs']):
                regret[label][i, :] = clairvoyant_logs[i, :] - \
                    logs[label][i, :]

        # inst reward, inst regret and cumulative regret plot
        x = np.arange(1, param_dict["horizon"]+1, step=250)
        f, ax = plt.subplots(3, figsize=(20, 30))
        sqrtn = np.sqrt(param_dict['n_epochs'])

        ax[0].plot(x, np.mean(clairvoyant_logs.T, axis=1)
                   [x], label='Clairvoyant', color='C0')
        ax[0].fill_between(x, np.mean(clairvoyant_logs.T, axis=1)[x]-np.std(clairvoyant_logs.T, axis=1)[x]/sqrtn,
                           np.mean(clairvoyant_logs.T, axis=1)[x]+np.std(clairvoyant_logs.T, axis=1)[x]/sqrtn, alpha=0.3, color='C0')
        for i, label in enumerate(regret.keys()):
            ax[0].plot(x, np.mean(logs[label].T, axis=1)
                       [x], label=label, color=f'C{i+1}')
            ax[0].fill_between(x, np.mean(logs[label].T, axis=1)[x]-np.std(logs[label].T, axis=1)[x]/sqrtn,
                               np.mean(logs[label].T, axis=1)[x]+np.std(logs[label].T, axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
            ax[1].plot(x, np.mean(regret[label].T, axis=1)
                       [x], label=label, color=f'C{i+1}')
            ax[1].fill_between(x, np.mean(regret[label].T, axis=1)[x]-np.std(regret[label].T, axis=1)[x]/sqrtn,
                               np.mean(regret[label].T, axis=1)[x]+np.std(regret[label].T, axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
            ax[2].plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[
                       x], label=label, color=f'C{i+1}')
            ax[2].fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn,
                               np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')

        ax[0].set_xlim(left=0)
        ax[0].set_title('Instantaneous Rewards')
        ax[0].legend()

        ax[1].set_xlim(left=0)
        ax[1].set_title('Instantaneous Regret')
        ax[1].legend()

        ax[2].set_xlim(left=0)
        ax[2].set_title('Cumulative Regret')
        ax[2].legend()

        # tikz.save(out_folder + f"tex/{testcase_id}_all.tex")
        plt.savefig(out_dir + f"png/{testcase}_all.png")

        #  cumulative regret plot
        x = np.arange(1, param_dict["horizon"]+50, step=50)
        x[-1] = min(x[-1],
                    len(np.mean(np.cumsum(regret['UCB1Agent'].T, axis=0), axis=1))-1)
        f, ax = plt.subplots(1, figsize=(20, 10))
        sqrtn = np.sqrt(param_dict['n_epochs'])

        for i, label in enumerate(regret.keys()):
            ax.plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[
                    x], label=label, color=f'C{i+1}')
            ax.fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn,
                            np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title('Cumulative Regret')
        ax.legend()

        # tikz.save(out_folder + f"tex/{testcase_id}_regret.tex")
        plt.savefig(out_dir + f"png/{testcase}_regret.png")

        # logging
        final_logs[f'{testcase}'] = {label: np.mean(
            np.sum(regret[label].T, axis=0)) for label in regret.keys()}

        # arm history plots
        n_arms = param_dict["n_arms"]
        f, ax = plt.subplots(3, 2, figsize=(20, 30))

        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_arms+1) - 0.5
            ax_.hist(a_hists[label].flatten(), bins=bins)
            ax_.set_xticks(range(n_arms))
            ax_.set_xlim(-1, n_arms)
            ax_.set_title(label)

        # tikz.save(out_dir + f"tex/{testcase}_a_hist.tex")
        plt.savefig(out_dir + f"png/{testcase}_a_hist.png")

        f, ax = plt.subplots(3, 2, figsize=(20, 30))
        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_arms+1) - 0.5
            ax_.plot(a_hists[label][-1, :])
            ax_.set_title(label)

        # tikz.save(out_folder + f"tex/{testcase_id}_a_hist_temp.tex")
        plt.savefig(out_dir + f"png/{testcase}_a_hist_temp.png")

    out_file = open(out_dir + f"logs.json", "w")
    json.dump(final_logs, out_file, indent=4)
    out_file.close()
