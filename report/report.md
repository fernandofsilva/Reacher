# Banana Collector using a Deep Q-Learning

## Introduction

the project environment, there are 20 identical copies of the agent. It has been shown that having multiple copies of the same agent sharing experience can accelerate learning, and you'll discover this for yourself when solving the project!

![](/images/arms.gif)

## Learning algorithm
For the task, it was used the Deep Deterministic Policy Gradient (DDPG) algorithm. 

### Parameters table

Below the table with the parameters used by the agent

| Parameter     | Value     | 
| ------------- |:---------:| 
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

The results show the reward per episode, the dots represent the reward per episode, the straight red line represents the reward moving average with the windows of 2 rewards, the dashed red line shows when the environment was solved (when the mean of the last 100 rewards are above 30), the dashed green line shows the mean of all rewards.

![](/images/scores.png)


## Future work

In future work, it would be an options to implement prioritized experience replay and others models like PPO, but due the results of the DDPG, it will be hard to find other model could beat it. 
