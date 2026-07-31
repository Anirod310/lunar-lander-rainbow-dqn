import torch
import torch.optim as optim
import torch.nn.functional as F
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

    def soft_update_target_network(self):
        for (target_param, local_param) in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data+(1-self.tau)*target_param.data)

    def learn(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor):
        q_values = self.q_network(states_tensor)
        current_q_values = q_values.gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0].unsqueeze(1)

        target_q_values = rewards_tensor + (self.gamma*max_next_q_values*(1-dones_tensor))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.soft_update_target_network()


    def update(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = self.memory.sample(self.batch_size)
            self.learn(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)

