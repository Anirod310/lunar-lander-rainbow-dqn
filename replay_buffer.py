import  random
import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity :
            self.memory.append((state, action, reward, next_state, done))
        else :
            self.memory[self.position] = (state, action, reward, next_state, done)

        self.position = (self.position + 1 ) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

        state_tensor = torch.tensor(np.array(batch_state), dtype=torch.float32)
        action_tensor = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1)
        reward_tensor = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
        next_state_tensor = torch.tensor(np.array(batch_next_state), dtype=torch.float32)
        done_tensor = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def __len__(self):
        return len(self.memory)






