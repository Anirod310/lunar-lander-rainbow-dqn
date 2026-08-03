# Project Report : Solving Lunar Lander with Rainbow DQN

**Author** : Dorian Bousseksou

**Project** : lunar-lander-rainbow-dqn

## Introduction 

Following my project on CartPole, I now decided to focus on the model described as the big boss of Q-Learning : the Rainbow DQN, which is a combination of several tweaks and improvements done to the standard DDQN to make it close to perfection in its domain.

The goal of this project was to solve the LunarLander-v3 environment. In this game, an agent must control a ship in order to make it land between a specific zone (marked out by flags), and thus solve the environment. The state consists of 8 continuous numbers : the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not. The four discrete actions the agent can do are : do nothing(0), fire left orientation engine(1), fire main engine(2), fire right orientation engine(3).

This time, the paper I studied to understand how this algorithm is implemented was ["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/pdf/1710.02298) by DeepMind. It allowed me to first focus more on how each of the Rainbow components works separately, as well as the mathematical foundations and intuition behind each of them.

## How The Agent Learns

### Mathematical Foundations

To build the Rainbow DQN, I combined six independent extensions of the standard DQN to create a state-of-the-art agent. Here is the mathematical intuition and the core formula behind each component:

#### 1. Double DQN (DDQN)
Standard DQN suffers from an overestimation bias because the same network is used to select and evaluate the best next action. DDQN separates this process by using the main network ($\theta$) to select the action, and the target network ($\bar{\theta}$) to evaluate its value : 
$$Y_t^{DoubleQ} = R_{t+1} + \gamma q_{\bar{\theta}}(S_{t+1}, \arg\max_{a'} q_\theta(S_{t+1}, a'))$$

#### 2. Dueling Networks
This architecture modifies the internal structure of the neural network. Instead of directly predicting the Q-values, the network splits into two separate streams: one estimating the global State Value $V(S)$ and the other estimating the Advantage $A(S, A)$ of each specific action. They are combined at the final output layer to compute the Q-values : 
$$Q(S, A) = V(S) + \left( A(S, A) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(S, a') \right)$$

#### 3. Prioritized Experience Replay (PER)
Instead of sampling transitions uniformly at random from the Replay Buffer as I did before, PER samples them based on the magnitude of their error (the Temporal Difference error, or KL divergence in a distributional context (that I will apply later)). This forces the agent to learn more frequently from unexpected outcomes where it made the biggest mistakes : 
$$p_i \propto |\delta_i|^\alpha$$
*(Where $p_i$ is the priority of the transition $i$, $\delta_i$ is the error, and $\alpha$ determines how much prioritization is used).*

#### 4. Multi-step Learning (n-step returns)
Instead of computing the error using only the immediate single next step, the agent looks $n$ steps ahead (we used $n=3$ because it proved to be well-balanced.) to accumulate the rewards before bootstrapping with the target network. This propagates the reward signals much faster through the network : 
$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a'} q_{\bar{\theta}}(S_{t+n}, a')$$

#### 5. Noisy Nets
Standard exploration techniques like $\epsilon$-greedy are replaced by adding parametric noise directly to the weights of the network's linear layers. The network can thus learn to ignore the noise (to exploit) or use it (to explore) depending on the state's complexity : 
$$y = (b + Wx) + (b_{noisy} \odot \epsilon_b + (W_{noisy} \odot \epsilon_W)x)$$

#### 6. Distributional RL (Categorical DQN / C51)
Instead of predicting a single expected average score for an action, the network predicts a categorical probability distribution of possible returns across a fixed set of atoms (51 buckets in the paper). The loss function becomes the Cross-Entropy (or KL Divergence) between the predicted distribution $\hat{Z}$ and the target distribution $Z$ projected via an operator $\Phi$ (which allows all the data to fits in a bucket) : 
$$\text{Loss} = D_{KL}(\Phi \hat{Z} || Z)$$

---

### First Code Architecture & Implementation Details

To translate this into code, the project is structured with different Python modules. In a first step, I only implemented the DDQN and the Dueling Network as follows : 

#### 1. Dueling Network Architecture (`model.py`)
Implemented as a PyTorch `nn.Module`, the model features a Dueling Q-Network structure:
* **Feature Extractor:** A shared fully-connected linear layer mapping the 8-dimensional observation vector to hidden representations (128 units, ReLU activation).
* **Value Stream $V(S)$:** A dedicated linear branch computing a scalar state-value estimate.
* **Advantage Stream $A(S, A)$:** Another linear branch computing relative advantage scores for each of the 4 discrete actions.
* **Aggregation Layer:** Combines streams using mean-subtracted advantage aggregation to ensure identifiability

#### 2. Experience Replay Buffer (`replay_buffer.py`)
A memory buffer storing transition tuples $(S_t, A_t, R_t, S_{t+1}, \text{done}_t)$ with random batch sampling. Uniform sampling breaks temporal correlations between consecutive environment frames to stabilize gradient descent. The next step consists in replace this simple architecture with a **Prioritized Experience Replay Buffer**, which focuses on the transitions where the agent made the biggest mistakes instead of picking memories at random.

#### 3. Agent & Loss Computation (`agent.py`)
The `Agent` class contains decision-making and learning functions:
* **Action Selection (`select_action`):** Implements an $\epsilon$-greedy exploration strategy.
* **Double DQN Target & Loss (`learn`):** Uses the local network to select optimal future actions and the target network to evaluate them, mitigating overestimation bias. The loss is optimized via Mean Squared Error (MSE):
  $$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( R + \gamma Q_{\bar{\theta}}\left(S', \arg\max_{a'} Q_\theta(S', a')\right) (1 - \text{done}) - Q_\theta(S, A) \right)^2 \right]$$
* **Soft Target Network Update (`soft_update_target_network`):** Smooths target network weight updates at every training step using Polyak averaging : ($\tau = 0.001$):
  $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{local}} + (1 - \tau) \theta_{\text{target}}$$

#### 4. Training and Evaluation Pipeline
The training process (`train.py`) follows a classic trial-and-error loop. The agent interacts with the `LunarLander-v3` environment over hundreds of episodes. As it plays, it progressively reduces its random exploration ($\\epsilon$-decay) to rely more on its learned neural network. Whenever the agent achieves a new high score, it automatically saves its "brain" (the network weights) to disk.

For the evaluation phase (`evaluate.py`), I loaded this best-performing brain and test the agent in a purely deterministic mode (zero random exploration). I also enabled the human render mode to visually watch the lander's flight dynamics and confirm its mastery of the environment.

**Results & Sample Efficiency** : The results of this current architecture (Dueling DDQN) are excellent. The training clearly followed three distinct phases:
1. **Crash:** Initially, the lander struggled, flipped, and crashed frequently.
2. **Survivial:** The agent learned to fire its main engine to slow its descent, avoiding catastrophic crashes but often missing the landing pad or drifting away.
3. **Mastery:** Eventually, the architecture converged really well. The agent learned to stabilize its angle using side thrusters, glide towards the flags, and perform soft landings, consistently scoring over **+200 points** (the official threshold for solving the environment). 

The baseline model solves the environment in 68705 timesteps. To measure our algorithm's performance, I used this metric as my primary benchmark for sample efficiency. my main goal was to drastically reduce this step count by incrementally implementing the advanced features of the Rainbow DQN.

*Note : To ensure fair comparison and reproducibility, all models were trained using the same fixed random seed(42).*

To see these results yourself, you just have to run `python evaluate.py` and see the agent lands by itself. 

### Adding Prioritized Experience Replay Buffer
While the current agent performs exceptionally well, the ultimate goal of this project is to implement a full **Rainbow DQN**. To reach this state-of-the-art architecture, the next iterations of the codebase will upgrade the current baseline with the **Prioritized Experience Replay Buffer**, in order to optimize the learning on the big mistakes the agent made.

To optimize learning, the standard uniform replay buffer was replaced with a **Prioritized Experience Replay (PER)** system. Instead of treating all past experiences equally, the agent now prioritizes "surprising" transitions where it made the biggest prediction errors, learning from its most critical mistakes first.

**Key Mechanisms:**
* **SumTree Data Structure:** To efficiently sample memories based on their priorities without slowing down the training loop, a binary `SumTree` was implemented from scratch. The leaves store the individual priorities, while the parent nodes store the sum of their children, allowing stratified sampling.
* **TD Error Prioritization:** After each neural network update, the absolute Temporal Difference (TD) error is extracted and used to update the memory's priority in the tree. A hyperparameter $\alpha = 0.6$ is used to smooth out these priorities and prevent the network from overfitting.
* **Importance Sampling (IS) Weights:** Because I intentionally bias the agent's perception of the environment (by forcing it to review its crashes and successes much more often than standard hovering), I had to correct the gradient updates. An IS weight (initially controlled by $\beta = 0.4$) is calculated for each sampled memory and multiplied directly into the Mean Squared Error loss.

**Results & Sample Efficiency (PER vs Baseline):**
With the PER architecture fully integrated, the agent no longer wastes time training on perfectly understood states. This targeted learning should, in theory, accelerate its understanding of the physical dynamics. However, after training both, I obtained these results : 
* **Previous Baseline (Dueling DDQN):** 68,654 timesteps to solve.
* **New Performance (with PER):** 75550 timesteps to solve.

This result, consistent with existing literature, demonstrates that prioritization alone in a highly stochastic environment causes overfitting to extreme errors. This justifies the necessity of coupling PER with N-Step Learning to restore temporal context to these errors, in order to fully benefit from prioritization and achieve significantly better results than the simple Replay Buffer.

### Making PER consistant by adding N-Step Learning 
The performance drop observed in the previous section showed a real limitation of using PER in isolation. To make prioritization consistent and truly effective, I implemented the N-Step Learning mechanism. Instead of learning from immediate transitions, the agent now waits for $N$ steps before computing its TD error. This forces the memory to prioritize meaningful sequences of actions rather than isolated incidents.

**Key Mechanisms:**
In traditional Q-learning, the agent updates its knowledge based on a single step, combining the immediate reward with the estimated value of the next state. N-step learning does it differently. Instead of looking just one step ahead, the agent accumulates real rewards over $N$ consecutive steps before bootstrapping the remaining value from the state reached at step $N$. This approach allows the agent to learn from delayed rewards much faster and more efficiently. The information about successful or catastrophic actions propagates quicker through the neural network, significantly accelerating the learning process while keeping the variance of the updates manageable.

**Results & Sample Efficiency:**
"After implementing N-Step Learning ($N=3$) and running the training on the same fixed seed, the agent solved the environment in 82,900 timesteps. While it did not beat the pure DDQN + Dueling Network baseline (68,705 steps), it showed a massive 25% improvement over the isolated PER architecture (110,000 steps).This result proves that multi-step returns successfully mitigate the PER's overfitting issue by restoring temporal context. However, it also highlights that on environments with dense reward signals like LunarLander, the variance introduced by N-Step and PER can sometimes outweigh their benefits compared to simpler models. To truly unlock this architecture's potential, we must continue stacking the remaining Rainbow components, starting with the network architecture itself."