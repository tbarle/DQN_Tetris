from collections import namedtuple
import torch

Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


def extract_tensor(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4

