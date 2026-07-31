import torch
import torch.optim as optim
from model import Rainbow
from replay_buffer import ReplayBuffer
import random

class Agent():
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.action_dim = action_dim
        self.q_network = Rainbow(state_dim, action_dim)
        self.target_network = Rainbow(state_dim, action_dim)

        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=config["learning_rate"])

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = ReplayBuffer(capacity=config["buffer_capacity"])

        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.epsilon = config["epsilon_start"]

    def select_action(self, state):
        random_number = random.random()
        if random_number < self.epsilon :
            return random.randint(0, self.action_dim-1)
        else :
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                best_q_value = torch.argmax(q_values).item()
                return best_q_value
