from collections import deque, namedtuple
from itertools import count
import math
import random
import torch

from DQN import deep_q_network
from main import GridWorld

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



batch_size = 64
gamma = 0.99 #discount factor
# epsilon-greedy parameters that will decrease over time
eps_start = .9
eps_end = .05
eps_decay = 2500*5
tau = 0.005 #target network update rate
learning_rate = 1e-3

nb_actions = 4
state_size = 2

policy_net = deep_q_network(state_size, nb_actions).to(device)
target_net = deep_q_network(state_size, nb_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # copy the weights from policy_net to target_net

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
memory = ReplayMemory(100)

step = 0

def select_action(state):
    """
    Select an action based on epsilon-greedy policy.
    
    :param state: Description
    """
    global step
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.*step/eps_decay) # the threshold will decrease over time
    step += 1
    if sample > eps_threshold: #select action based on policy
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1,1)
    else: #select random action for exploration
        return torch.tensor([[random.randrange(nb_actions)]], device=device, dtype=torch.long)



def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Identify non-terminal states
    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s,a) for all states
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # compute V(s') = max_a' Q(s',a') only for non-terminal states
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # expected Q values: r + gamma * V(s') for non-terminal, r for terminal
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # compute Loss
    criterion = torch.nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize the model using backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(env:GridWorld):
    try:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        for i_episode in range(num_episodes):
            # Reset environment at the start of each episode
            env.reset()
            state = torch.tensor(env.get_state(), dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            
            for t in count():
                # state = torch.tensor(env.get_random_state(), dtype=torch.float32, device=device).unsqueeze(0)
                action = select_action(state)
                observation, reward = env.move(action.item())
                # observation, reward = env.get_state(), env.get_reward()
                reward = torch.tensor([reward], device=device)
                episode_reward += reward.item()

                # next_state is None if episode terminates (goal reached or max steps)
                done = env.is_goal() or t >= 99  # max 100 steps per episode
                # done = t >=99
                next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state if next_state is not None else state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    break
            
            if (i_episode + 1) % 50 == 0:
                print(f'Episode {i_episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}')
                print(f'Epsilon: {eps_end + (eps_start - eps_end) * math.exp(-1.*step/eps_decay):.4f}')

        print('Training Complete')
    finally:
        return policy_net
