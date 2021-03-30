# Banana Collector using a Deep Q-Learning

## Introduction

the project environment, there are 20 identical copies of the agent. It has been shown that having multiple copies of the same agent sharing experience can accelerate learning, and you'll discover this for yourself when solving the project!

![](/images/arms.gif)

## Deep Deterministic Policy Gradient (DDPG)

It was used the DDPG algorithm for the task. Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. 
It uses off-policy data, and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way: if you know the optimal action-value function $Q^{*}(s,a)$, then in any given state, the optimal action $a^{*}(s)$ can be found by solving
$$a^{*}(s)=argmaxQ^{*}(s,a)$$

DDPG interleaves learning an approximator to $Q^{*}(s,a)$ with learning an approximator to $a^{*}(s)$. Because the action space is continuous, 
the function $Q^{*}(s,a)$ is presumed to be differentiable with respect to the action argument. It allows us to set up an efficient, 
gradient-based learning rule for a policy $\mu(s)$ which exploits that fact. 
Then, instead of running an expensive optimization subroutine each time we wish to compute $\max_a Q(s,a)$, we can approximate it with \max_a Q(s,a) \approx Q(s,\mu(s)) [(OpenAI, 2018)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).

### Parameters table

Below the table with the parameters used by the agent

| Parameter     | Value     | 
| --------------|:---------:| 
| n_episodes    | 200       |
| buffer_size   | int(1e5)  |
| batch_size    | 128       |
| gamma         | 0.99      |
| tau           | 1e-3      |
| lr_actor      | 1e-4      |
| lr_critc      | 1e-4      |
| weight_decay  | 0         |

### Neural network

It was used a neural network with 2 hidden layers with 256 and 128 nodes each one, all hidden layers used ReLu as activation function.
The output layer has 4 nodes (according the action space 0, 1, 2 and 3) with tahn activation function (output between -1 and 1).

![](/images/nn.svg)


## Results

The results show the reward per episode, the dots represent the reward per episode, the straight red line represents the reward moving average with the windows of 2 rewards, 
the dashed red line shows when the environment was solved (when the mean of the last 100 rewards are above 30), the dashed green line shows the mean of all rewards.

The scores of each episode correspond to the mean of the rewards of the 20 agents (version 2), the environment was solved in the episode 104,
in other words, the mean of the episode 4 to 104 ware above 30. 

![](/images/scores.png)


## Future work

In future work, it would be an options to implement prioritized experience replay and others models like PPO, but due the results of the DDPG, it will be hard to find other model could beat it. 



