import  random
import torch
import numpy as np
from sum_tree import SumTree

class ReplayBuffer:
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-5

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        data = (state, action, reward, next_state, done)

        self.tree.add(max_priority, data)

    def sample(self, batch_size, beta=0.4):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)

        total_priority = self.tree.total_priority
        segment_size = total_priority / batch_size

        for i in range(batch_size):
            segment_boundaries_a = i * segment_size
            segment_boundaries_b = (i + 1) * segment_size

            v = random.uniform(segment_boundaries_a, segment_boundaries_b)

            index, priority, data = self.tree.get_leaf(v)

            probability = priority / total_priority

            weight = (self.capacity * probability) ** (-beta)
            weights[i] = weight

            indices[i] = index

            state, action, reward, next_state, done = data

            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_dones.append(done)

        weights /= weights.max()

        states_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32)
        actions_tensor = torch.tensor(batch_actions, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(batch_next_states), dtype=torch.float32)
        dones_tensor = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1)

        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor), indices, torch.FloatTensor(weights).unsqueeze(1)

    def update_priorities(self, indices, td_errors):
        for i in range(len(indices)):
            new_priority = (abs(td_errors[i].item()) + self.epsilon) ** self.alpha
            self.tree.update(indices[i], new_priority)






