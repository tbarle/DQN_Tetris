from itertools import count

import torch
from torch import optim
import torch.nn.functional as F

from TetrisENV import TetrisEnvironment
from EpsilonGreedyStrategy import EpsilonGreedyStrategy
from ReplayMemory import ReplayMemory
from DeepQNetwork import DQN
from Agent import Agent
from Experience import Experience, extract_tensor


batch_size = 256
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TetrisEnvironment()
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
replay_memory = ReplayMemory(memory_size)

policy_net = DQN().to(device)
target_net = DQN().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_scores = []

for episode in range(num_episodes):
    env.reset()
    board = env.get_current_board_state()
    state = env.get_state_properties(board)

    next_states_dict = env.get_next_states()
    next_actions, next_states = zip(*next_states_dict.items())
    print(next_actions)
    agent = Agent(strategy, next_actions, next_states_dict)

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward, done = env.step(action)     #Eseguo l'azione

        board = env.get_current_board_state()
        next_state = env.get_state_properties(board)
        replay_memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if replay_memory.can_provide_sample(batch_size):
            experiences = replay_memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensor(experiences)

            #policy_net.eval()
            current_q_values = policy_net(states)
            next_q_values = target_net(next_states)
            target_q_values = (gamma * next_q_values) + rewards
            loss = F.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            #final_score = env.score
            #final_tetrominoes = env.tetrominoes
            #final_cleared_lines = env.cleared_lines
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

