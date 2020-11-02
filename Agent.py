import random
import torch


class Agent:
    def __init__(self, strategy, next_actions, next_states_dict):
        self.strategy = strategy
        self.current_step = 0
        self.next_actions = next_actions
        self.next_states_dict = next_states_dict

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            index = random.randint(0, len(self.next_states_dict) - 1)
            print('Random')
        else:
            with torch.no_grad():
                index = policy_net(state).argmax().item()
            print('Not random')
        return self.next_actions[index]
