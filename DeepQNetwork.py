import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(in_features=4, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        #x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
