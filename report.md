# Project Report : Solving Lunar Lander with Rainbow DQN

**Author** : Dorian Bousseksou

**Project** : lunar-lander-rainbow-dqn

## Introduction 

Following my project on CartPole, I now decided to focus on the model described as the big boss of Q-Learning : the Rainbow DQN, which is a combination of several tweaks and improvements done to the standard DDQN to make it close to perfection in its domain.

The goal of this project was to solve the LunarLander-v3 environment. In this game, an agent must control a ship in order to make it land between a specific zone (marked out by flags), and thus solve the environment. The state consists of 8 continuous numbers : the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not. The four discrete actions the agent can do are : do nothing(0), fire left orientation engine(1), fire main engine(2), fire right orientation engine(3).

This time, the paper I studied to understand how this algorithm is implemented was ["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/pdf/1710.02298) by DeepMind. It allowed me to first focus more on how each of the Rainbow components works separately, as well as the mathematical foundations and intuition behind each of them.

## How The Agent Learns

### Mathematical Foundations

To build the Rainbow DQN, we combine six independent extensions of the standard DQN to create a state-of-the-art agent. Here is the mathematical intuition and the core formula behind each component:

#### 1. Double DQN (DDQN)
Standard DQN suffers from an overestimation bias because the same network is used to select and evaluate the best next action. DDQN decouples this process by using the main network ($\theta$) to select the action, and the target network ($\bar{\theta}$) to evaluate its value.
$$Y_t^{DoubleQ} = R_{t+1} + \gamma q_{\bar{\theta}}(S_{t+1}, \arg\max_{a'} q_\theta(S_{t+1}, a'))$$

#### 2. Prioritized Experience Replay (PER)
Instead of sampling transitions uniformly at random from the Replay Buffer, PER samples them based on the magnitude of their error (the Temporal Difference error, or KL divergence in a distributional context). This forces the agent to learn more frequently from unexpected outcomes where it made the biggest mistakes.
$$p_i \propto |\delta_i|^\alpha$$
*(Where $p_i$ is the priority of the transition $i$, $\delta_i$ is the error, and $\alpha$ determines how much prioritization is used).*

#### 3. Dueling Networks
This architecture modifies the internal structure of the neural network. Instead of directly predicting the Q-values, the network splits into two separate streams: one estimating the global State Value $V(S)$ and the other estimating the Advantage $A(S, A)$ of each specific action. They are aggregated at the final output layer to compute the Q-values.
$$Q(S, A) = V(S) + \left( A(S, A) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(S, a') \right)$$

#### 4. Multi-step Learning (n-step returns)
Instead of computing the error using only the immediate single next step, the agent looks $n$ steps ahead (e.g., $n=3$) to accumulate the rewards before bootstrapping with the target network. This propagates the reward signals much faster through the network.
$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a'} q_{\bar{\theta}}(S_{t+n}, a')$$

#### 5. Distributional RL (Categorical DQN / C51)
Instead of predicting a single expected average score for an action, the network predicts a categorical probability distribution of possible returns across a fixed set of atoms (e.g., 51 buckets). The loss function becomes the Cross-Entropy (or KL Divergence) between the predicted distribution $\hat{Z}$ and the target distribution $Z$ projected via an operator $\Phi$.
$$\text{Loss} = D_{KL}(\Phi \hat{Z} || Z)$$

#### 6. Noisy Nets
Standard exploration techniques like $\epsilon$-greedy are replaced by adding parametric noise directly to the weights of the network's linear layers. The network can thus learn to ignore the noise (to exploit) or use it (to explore) depending on the state's complexity.
$$y = (b + Wx) + (b_{noisy} \odot \epsilon_b + (W_{noisy} \odot \epsilon_W)x)$$