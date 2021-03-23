from unityagents import UnityEnvironment
from monitor import interact
from agent import Agent
import argparse

# Instantiate argument parser
parser = argparse.ArgumentParser()

# Add arguments (Interaction with the environment)
parser.add_argument('--n_episodes', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--max_t', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--eps_start', nargs='?', const=1, type=float, default=1.0)
parser.add_argument('--eps_end', nargs='?', const=1, type=float, default=0.01)
parser.add_argument('--eps_decay', nargs='?', const=1, type=float, default=0.995)

# Add arguments (Agent)
parser.add_argument('--buffer_size', nargs='?', const=1, type=int, default=int(1e5))
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=128)
parser.add_argument('--gamma', nargs='?', const=1, type=float, default=0.99)
parser.add_argument('--tau', nargs='?', const=1, type=float, default=1e-3)
parser.add_argument('--lr_actor', nargs='?', const=1, type=float, default=1e-4)
parser.add_argument('--lr_critic', nargs='?', const=1, type=float, default=1e-3)
parser.add_argument('--weight_decay', nargs='?', const=1, type=float, default=1e-3)

# Parser parameters
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)

# Pass args
args = parser.parse_args()


if __name__ == '__main__':

    # Create environment
    # env = UnityEnvironment(file_name="Reacher_Linux_NoVis/Reacher.x86_64")
    env = UnityEnvironment(file_name="unity/Reacher_Linux_NoVis/Reacher.x86_64")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Instantiate agent
    agent = Agent(
        state_size=len(env_info.vector_observations[0]),
        action_size=brain.vector_action_space_size,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        weight_decay=args.weight_decay,
        seed=0
    )

    # Interact with environment
    scores = interact(
        env,
        agent,
        brain_name=brain_name,
        n_episodes=args.n_episodes,
        max_t=args.max_t,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay
    )














# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))




