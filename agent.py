import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Rainbow
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import random

class Agent:
    def __init__(self, state_dim, action_dim, n_atoms, config):
        self.config = config
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.q_network = Rainbow(state_dim, action_dim, n_atoms)
        self.target_network = Rainbow(state_dim, action_dim, n_atoms)

        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=config["learning_rate"])

        self.target_network.load_state_dict(self.q_network.state_dict())

        #self.memory = ReplayBuffer(config["buffer_capacity"])

        self.memory = PrioritizedReplayBuffer(
            config["buffer_capacity"],
            state_dim,
            action_dim,
            alpha=0.6
        )

        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        #self.epsilon = config["epsilon_start"] --> Unused since I implemented noisy linear

    def select_action(self, state):

        self.q_network.reset_all_noise()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.q_network(state_tensor)
            expected_value = torch.sum(self.q_network.Z_tensor * probs, dim=-1)
            return torch.argmax(expected_value, dim=-1).item()

        #All the part below is relevant only with epsilon greedy exploration method : 
        '''random_number = random.random()
        if random_number < self.epsilon :
            return random.randint(0, self.action_dim-1)
        else :
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                best_q_value = torch.argmax(q_values).item()
                return best_q_value'''

    def soft_update_target_network(self):
        for (target_param, local_param) in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data+(1-self.tau)*target_param.data)

    def learn(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, indices, weights ):

        self.q_network.reset_all_noise()
        self.target_network.reset_all_noise()

        with torch.no_grad():

            next_probs = self.q_network(next_states_tensor)

            expected_values  = torch.sum(next_probs * self.q_network.Z_tensor, dim=-1)

            best_future_action_index = torch.argmax(expected_values, dim=-1, keepdim=True)

            target_next_probs = self.target_network(next_states_tensor)

            expended_indices = best_future_action_index.unsqueeze(-1).expand(-1, 1, self.n_atoms)

            best_next_probs = torch.gather(target_next_probs, dim=1, index= expended_indices).squeeze(1)

            n_step = self.memory.n_step

            gamma_n = self.gamma ** n_step

            Tz = (self.q_network.Z_tensor * gamma_n * (1 - dones_tensor)) + rewards_tensor
            Tz = torch.clamp(Tz, self.q_network.Z_tensor[0], self.q_network.Z_tensor[-1])

            delta_z = (self.q_network.Z_tensor[-1]-self.q_network.Z_tensor[0]) / (self.n_atoms-1)

            b = (Tz-self.q_network.Z_tensor[0]) / delta_z

            lower = torch.clamp(b.floor().long(), 0, self.n_atoms-1)
            upper = torch.clamp(b.ceil().long(), 0, self.n_atoms-1)

            m = torch.zeros_like(best_next_probs)

            for j in range(self.n_atoms):

                l = lower[:, j].unsqueeze(1)
                u = upper[:, j].unsqueeze(1)

                l_part = (best_next_probs[:, j] * (upper[:, j].float() - b[:, j])).unsqueeze(1)
                u_part = (best_next_probs[:, j] * (b[:, j] - lower[:, j].float())).unsqueeze(1)

                exact_match = (upper[:, j] == lower[:, j]).unsqueeze(1)
                l_part = torch.where(exact_match, best_next_probs[:, j].unsqueeze(1), l_part)
                u_part = torch.where(exact_match, torch.zeros_like(u_part), u_part)

                m.scatter_add_(1, l, l_part)
                m.scatter_add_(1, u, u_part)

        current_probs = self.q_network(states_tensor)
        actions_expended = actions_tensor.unsqueeze(-1).expand(-1, 1, self.n_atoms)
        current_action_probs = torch.gather(current_probs, dim=1, index=actions_expended).squeeze(1)

        loss_elementwise = - ( m * torch.log(current_action_probs + 1e-8)).sum(dim=1)

        loss = (loss_elementwise * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        self.optimizer.step()

        self.soft_update_target_network()

        td_errors = loss_elementwise.detach().cpu().numpy() + 1e-5

        self.memory.update_priorities(indices, td_errors)


    def update(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            batch, indices, weights = self.memory.sample(self.batch_size, beta=0.4)
            states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = batch
            self.learn(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, indices, weights)

