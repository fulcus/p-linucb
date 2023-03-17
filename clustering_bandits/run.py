from src.agents import *
from src.CLUB import CLUB
from src.environment import PartitionedEnvironment
from src.core import Core
from src.utils import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import json
import os
import time

os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim_id', default=0)
    parser.add_argument('-t', '--test_file', default=None)
    args = parser.parse_args()

    in_dir = 'clustering_bandits/test/input/'
    out_dir = f'clustering_bandits/test/output/simulation_{args.sim_id}/'
    # os.makedirs(out_dir + 'tex/', exist_ok=True)
    os.makedirs(out_dir + 'png/', exist_ok=True)
    final_logs = {}
    test_files = os.listdir(
        in_dir) if args.test_file is None else [args.test_file]

    start_all_time = start_time = time.time()
    for testcase in test_files:
        print(f'############## {testcase} ##############')

        with open(f'{in_dir}{testcase}') as f:
            param_dict = json.load(f)
        testcase, _ = testcase.split('.')
        # list to np.ndarray
        for k, v in param_dict.items():
            if type(v) == list:
                param_dict[k] = np.squeeze(np.asarray(v))

        logs = {}
        a_hists = {}
        t_splits = {}

        env = PartitionedEnvironment(n_rounds=param_dict["horizon"],
                                     arms=param_dict["arms"],
                                     n_contexts=param_dict["n_contexts"],
                                     theta=param_dict["theta"],
                                     theta_p=param_dict["theta_p"],
                                     k=param_dict["k"],
                                     sigma=param_dict['sigma'],
                                     random_state=param_dict['seed'])

        # Clairvoyant
        agent = Clairvoyant(arms=param_dict["arms"],
                            n_contexts=param_dict["n_contexts"],
                            theta=param_dict["theta"],
                            theta_p=param_dict["theta_p"],
                            k=param_dict["k"])
        core = Core(env, agent)
        # rewards, arms
        clairvoyant_logs, a_hists['Clairvoyant'], _, _ = core.simulation(
            n_epochs=param_dict['n_epochs'], n_rounds=param_dict["horizon"])
        print(f"Clairvoyant - {time.time() - start_time:.2f}s")
        start_time = time.time()
        # Reward upper bound
        max_reward = clairvoyant_logs.max()

        agents_list = [
            # UCB1Agent(param_dict["arms"],
            #           param_dict["n_contexts"],
            #           max_reward),
            # LinUCBAgent(param_dict["arms"],
            #             param_dict["n_contexts"],
            #             param_dict["horizon"],
            #             1,
            #             param_dict["max_theta_norm_shared"] + param_dict["max_theta_norm_p"],
            #             param_dict["max_arm_norm"],
            #             param_dict['sigma']),
            # INDUCB1Agent(param_dict["arms"],
            #              param_dict["n_contexts"],
            #              max_reward),
            INDLinUCBAgent(
                param_dict["arms"],
                param_dict["n_contexts"],
                param_dict["horizon"],
                1,
                param_dict["max_theta_norm"],
                param_dict["max_arm_norm"],
                param_dict['sigma']),
            PartitionedAgentStatic(
                param_dict["arms"],
                param_dict["n_contexts"],
                param_dict["horizon"],
                1,
                param_dict["max_theta_norm"],
                param_dict["max_theta_norm_local"],
                param_dict["max_arm_norm"],
                param_dict["max_arm_norm_local"],
                k=param_dict["k"],
                err_th=1e-1,
                win=30,
                sigma=param_dict['sigma']),
            # PartitionedAgentConstrDyn(
            #     param_dict["arms"],
            #     param_dict["n_contexts"],
            #     param_dict["horizon"],
            #     1,
            #     param_dict["max_theta_norm"],
            #     param_dict["max_theta_norm_local"],
            #     param_dict["max_arm_norm"],
            #     param_dict["max_arm_norm_local"],
            #     k=param_dict["k"],
            #     err_th=1e-1,
            #     win=30,
            #     sigma=param_dict['sigma']),
            # PartitionedAgentDyn(
            #     param_dict["arms"],
            #     param_dict["n_contexts"],
            #     param_dict["horizon"],
            #     1,
            #     param_dict["max_theta_norm"],
            #     param_dict["max_theta_norm_local"],
            #     param_dict["max_arm_norm"],
            #     param_dict["max_arm_norm_local"],
            #     k=param_dict["k"],
            #     err_th=1e-1,
            #     win=30,
            #     sigma=param_dict['sigma']),
            # CLUB(param_dict["arms"],
            #      param_dict["n_contexts"],
            #      param_dict["horizon"]),
        ]

        # Train all agents
        for agent in agents_list:
            agent_name = agent.__class__.__name__
            core = Core(env, agent)
            if isinstance(agent, PartitionedAgentStatic):
                # for w in [10, 20, 30, 50, 70, 100]:
                #     agent_name = f"{agent.__class__.__name__}_{w}"
                #     agent.win = w
                #     core = Core(env, agent)
                #     logs[agent_name], a_hists[agent_name], t_splits[agent_name], err_hists = core.simulation(
                #         n_epochs=param_dict["n_epochs"], n_rounds=param_dict["horizon"])
                logs[agent_name], a_hists[agent_name], t_splits[agent_name], err_hists = core.simulation(
                    n_epochs=param_dict["n_epochs"], n_rounds=param_dict["horizon"])
            else:
                logs[agent_name], a_hists[agent_name], _, _ = core.simulation(
                    n_epochs=param_dict["n_epochs"], n_rounds=param_dict["horizon"])

            print(f"{agent_name} - {time.time() - start_time:.2f}s")
            start_time = time.time()
        print(f"Total training {time.time() - start_all_time:.2f}s")

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

        testcase_dir = out_dir + f'png/{testcase}/'
        os.makedirs(testcase_dir, exist_ok=True)
        # tikz.save(out_folder + f"tex/{testcase_id}_all.tex")
        plt.savefig(testcase_dir + "all.png")

        #  cumulative regret plot
        x = np.arange(1, param_dict["horizon"]+50, step=50)
        first_log = next(iter(regret.values()))
        x[-1] = min(x[-1],
                    len(np.mean(np.cumsum(first_log.T, axis=0), axis=1))-1)
        f, ax = plt.subplots(1, figsize=(20, 10))
        n_epochs = param_dict['n_epochs']
        sqrtn = np.sqrt(n_epochs)
        for i, label in enumerate(regret.keys()):
            # one plot per epoch
            for j in range(n_epochs):
                ax.plot(x, np.cumsum(regret[label][j].T, axis=0)[x],
                        label=f"{label} ep {j}", color=f'C{n_epochs+i+j+1}')
                if label.startswith("PartitionedAgent") and t_splits[label][j] is not None:
                    ax.axvline(
                        x=t_splits[label][j], color=f'C{n_epochs+i+j+1}', label=f'split time ep {j}')
            if n_epochs > 1:
                # mean and std
                ax.plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[
                        x], label=f"{label} mean", color=f'C{i+n_epochs+1}')
                ax.fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn,
                                np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+n_epochs+1}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title('Cumulative Regret')
        ax.legend()

        # tikz.save(out_folder + f"tex/{testcase_id}_regret.tex")
        plt.savefig(testcase_dir + "regret.png")

        # Error plots
        """
        if err_hists.any():
            # err_hists.shape = (n_epochs, n_contexts, horizon, 1)
            # n_epochs = err_hists.shape[0]
            n_contexts = err_hists.shape[1]
            
            err_logs = {}  # epoch: context: [errors at each timestep]
            # by epoch
            for i in range(n_epochs):
                err_logs[i] = {}
                f, ax = plt.subplots(1, figsize=(20, 10))
                for j in range(n_contexts):
                    err_logs[i][j] = err_hists.squeeze()[i, j].tolist()
                    ax.plot(x, err_hists[i, j][x],
                            label=f"c_{j}", color=f'C{j+1}')
                    if label == "PartitionedAgent" and t_splits[i] is not None:
                        ax.axvline(
                            x=t_splits[i], color=f'C{i+1}', label=f'split time')

                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.set_title(f'Squared Error, Epoch {i}')
                ax.legend()
                plt.savefig(testcase_dir + f"err_ep_{i}.png")

            with open(testcase_dir + "err.json", "w") as f:
                json.dump(err_logs, f, indent=4)

            # by context
            for j in range(n_contexts):
                f, ax = plt.subplots(1, figsize=(20, 10))
                for i in range(n_epochs):
                    ax.plot(x, err_hists[i, j][x],
                            label=f"ep {i}", color=f'C{i+1}')
                    if label == "PartitionedAgent" and t_splits[i] is not None:
                        ax.axvline(x=t_splits[i], color=f'C{i+1}',
                                label=f'split time ep {i}')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.set_title(f'Squared Error, Context {j}')
                ax.legend()
                plt.savefig(testcase_dir + f"err_c_{j}.png")
        """
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
            plt.savefig(testcase_dir + f"a_hist.png")

            f, ax = plt.subplots(3, 2, figsize=(20, 30))
            for ax_, label in zip(f.axes, a_hists.keys()):
                bins = np.arange(n_arms+1) - 0.5
                ax_.plot(a_hists[label][-1, :])
                ax_.set_title(label)

            # tikz.save(out_folder + f"tex/{testcase_id}_a_hist_temp.tex")
            plt.savefig(testcase_dir + f"a_hist_temp.png")

    with open(out_dir + f"logs.json", "w") as f:
        json.dump(final_logs, f, indent=4)
