## Partitioned Linear Bandit Algorithm

This repository contains the implementation of the Partitioned Linear Bandit (P-LinUCB) algorithm. P-LinUCB is designed to solve the partitioned linear bandit problem, a special case of the multi-agent linear bandit setting, where the bandit parameter consists of shared global components and local components specific to each bandit instance. This algorithm leverages this partitioning to learn the global components using data from all bandit instances, while learning the local components separately.

### Partitioned Linear Setting
The partitioned setting in the linear bandit problem involves decomposing the bandit parameter into shared and local components. The shared components are common across all bandits, while the local components are specific to each bandit instance. At each time step, an agent chooses an arm and observes a reward generated based on the shared and local components. The algorithm aims to learn both the shared and local components to optimize the cumulative reward.

### P-LinUCB Algorithm
The P-LinUCB algorithm starts with each bandit instance independently learning its own parameter. Once one of the bandit instances achieves a low enough regression error, indicating a reliable estimate of the shared parameter, the algorithm switches to updating only the local components for all agents. This approach speeds up the learning process by focusing on the local components after obtaining a good estimate of the shared parameter.

The main steps of the P-LinUCB algorithm are as follows:
1. Instantiate $N$ LinUCB instances, one for each bandit.
2. At each time step $t$, select a bandit index $i$.
3. If the aggregation has not occurred yet:
   - Choose an action based on the standard LinUCB algorithm for bandit $i$.
   - Receive the reward and update the parameters of bandit $i$.
   - Check the aggregation condition based on a defined criterion.
4. If the aggregation condition is met:
   - Split the parameters of all agents into global and local components.
   - Update only the local components for all agents.
5. Repeat steps 2-4 until the end of the time horizon $T$.

#### Setup
To use the P-LinUCB algorithm, the following inputs are required:
- $N$: Number of agents.
- $T$: Time horizon.
- $` \{\mathcal{A}_i\}_{i=1}^{N} `$: Set of arms for each agent.
- $k$: Partitioning parameter that splits the arms dimension into global and local components.
- $w$: Sliding window dimension for the aggregation criterion.
- $\lambda$: Regularization hyper-parameter.

The P-LinUCB algorithm instantiates $N$ LinUCB instances according to the standard LinUCB algorithm.

#### Update
The update of the bandit parameters follows the LinUCB algorithm. At each time step, a bandit index $i$ is sampled, and the action and update depend on whether the aggregation has occurred or not.

#### Aggregation Criterion
The aggregation criterion determines when the aggregation condition is met. It evaluates the regression error on the prediction of the reward. The Mean Absolute Percentage Error (MAPE) is used as the regression error metric. The criterion computes the MAPE over the last $w$ samples in a moving average fashion. If the moving average of the residuals falls below a threshold $\varepsilon$, indicating a good enough estimate of the parameter, the aggregation condition is satisfied.

#### Split
When the aggregation condition is satisfied, the algorithm saves the estimated optimal sub-arm and performs the split operation. The split partitions the parameters into global and local components for all agents. The local sub-arms become the focus for each agent's actions, while the global optimal sub-arm has been found. The local parameters are recomputed using a closed-form solution.

