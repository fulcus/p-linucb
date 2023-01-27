import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

if __name__ == '__main__':
    from src.agents import UCB1Agent, Clairvoyant, LinUCBAgent
    from src.environment import LinearEnvironment
    from src.core import Core
    import matplotlib.pyplot as plt
    import numpy as np
    import tikzplotlib as tikz
    import warnings
    import json
    import sys

    warnings.filterwarnings('ignore')
    final_logs = {}
    simulation_id = sys.argv[1]
    out_folder = f'clustering_bandits/test/output/simulation_{simulation_id}/'
    os.makedirs(out_folder + 'tex/', exist_ok=True)
    os.makedirs(out_folder + 'png/', exist_ok=True)

    for testcase_id in sys.argv[2:]:
        print(f'################## Testcase {testcase_id} ###################')

        f = open(f'clustering_bandits/test/input/testcase_{testcase_id}.json')
        param_dict = json.load(f)
        print(f'Parameters: {param_dict}')

        # list to np.array
        for k, v in param_dict.items():
            if type(v) == list:
                param_dict[k] = np.asarray(v)

        logs = {}
        a_hists = {}

        # Clairvoyant
        print('Training Clairvoyant algorithm')
        clrv = 'Clairvoyant'
        env = LinearEnvironment(n_rounds=param_dict["horizon"], theta=param_dict["theta"],
                                noise_std=param_dict['noise_std'], random_state=param_dict['seed'])
        agent = Clairvoyant(
            n_arms=param_dict["n_actions"], actions=param_dict["actions"], exp_reward=param_dict['exp_reward'])
        core = Core(env, agent)
        # rewards, actions
        clairvoyant_logs, a_hists['Clairvoyant'] = core.simulation(
            n_epochs=param_dict['n_epochs'], n_rounds=param_dict["horizon"])
        clairvoyant_logs = clairvoyant_logs[:, 1:]

        # Reward upper bound
        max_reward = clairvoyant_logs.max()

        # UCB1
        print('Training UCB1 Algorithm')
        env = LinearEnvironment(n_rounds=param_dict["horizon"], theta=param_dict["theta"],
                                noise_std=param_dict['noise_std'], random_state=param_dict['seed'])
        agent = UCB1Agent(param_dict["n_actions"],
                          param_dict["actions"], max_reward=max_reward)
        core = Core(env, agent)
        logs['UCB1'], a_hists['UCB1'] = core.simulation(
            n_epochs=param_dict["n_epochs"], n_rounds=param_dict["horizon"])
        logs['UCB1'] = logs['UCB1'][:, 1:]

        # LinUCB
        print('Training LinUCB Algorithm')
        env = LinearEnvironment(n_rounds=param_dict["horizon"], theta=param_dict["theta"],
                                noise_std=param_dict['noise_std'], random_state=param_dict['seed'])
        agent = LinUCBAgent(param_dict["actions"], param_dict["horizon"], lmbd=1,
                            max_theta_norm=param_dict["n_actions"], max_action_norm=1.5)
        core = Core(env, agent)
        logs['LinUCB'], a_hists['LinUCB'] = core.simulation(
            n_epochs=param_dict['n_epochs'], n_rounds=param_dict["horizon"])
        logs['LinUCB'] = logs['LinUCB'][:, 1:]

        # Regrets computing
        print('Computing regrets...')
        clairvoyant_logs = clairvoyant_logs.astype(np.float64)

        regret = {label: np.inf *
                  np.ones((param_dict['n_epochs'], param_dict["horizon"])) for label in logs.keys()}
        for i in range(param_dict['n_epochs']):
            for label in regret.keys():
                logs[label] = logs[label].astype(np.float64)
                regret[label][i, :] = clairvoyant_logs[i, :] - \
                    logs[label][i, :]

        # inst reward, inst regret and cumulative regret plot
        x = np.arange(1, param_dict["horizon"]+1, step=250)
        f, ax = plt.subplots(3, figsize=(20, 30))
        sqrtn = np.sqrt(param_dict['n_epochs'])

        ax[0].plot(x, np.mean(clairvoyant_logs.T, axis=1)
                   [x], label=clrv, color='C0')
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

        # tikz.save(out_folder + f"tex/testcase_{testcase_id}_all.tex")
        plt.savefig(out_folder + f"png/testcase_{testcase_id}_all.png")

        #  cumulative regret plot
        x = np.arange(1, param_dict["horizon"]+50, step=50)
        x[-1] = min(x[-1],
                    len(np.mean(np.cumsum(regret['UCB1'].T, axis=0), axis=1))-1)
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

        # tikz.save(out_folder + f"tex/testcase_{testcase_id}_regret.tex")
        plt.savefig(out_folder + f"png/testcase_{testcase_id}_regret.png")

        # logging
        final_logs[f'testcase_{testcase_id}'] = {label: np.mean(
            np.sum(regret[label].T, axis=0)) for label in regret.keys()}

        # action history plots
        n_actions = param_dict["n_actions"]
        f, ax = plt.subplots(3, 2, figsize=(20, 30))

        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_actions+1) - 0.5
            ax_.hist(a_hists[label].flatten(), bins=bins)
            ax_.set_xticks(range(n_actions))
            ax_.set_xlim(-1, n_actions)
            ax_.set_title(label)

        tikz.save(out_folder + f"tex/testcase_{testcase_id}_a_hist.tex")
        plt.savefig(out_folder + f"png/testcase_{testcase_id}_a_hist.png")

        f, ax = plt.subplots(3, 2, figsize=(20, 30))
        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_actions+1) - 0.5
            ax_.plot(a_hists[label][-1, :])
            ax_.set_title(label)

        # tikz.save(out_folder + f"tex/testcase_{testcase_id}_a_hist_temp.tex")
        plt.savefig(out_folder + f"png/testcase_{testcase_id}_a_hist_temp.png")

    out_file = open(out_folder + f"logs.json", "w")
    json.dump(final_logs, out_file, indent=4)
    out_file.close()