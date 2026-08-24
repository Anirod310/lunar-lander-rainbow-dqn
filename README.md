# Solving Lunar Lander with Rainbow DQN from Scratch

<p align="left">
  <img src="https://gymnasium.farama.org/_images/lunar_lander.gif" alt="Lunar Lander" width="300"/>
</p>

This project is the continuation of the previous one on the CartPole environment and the Deep Q-Network (DQN / DDQN) algorithm. In this project, I implemented the state-of-the-art value-based algorithm : the **Rainbow DQN**, which combines six foundational extensions of Deep Q-Learning from DeepMind's research paper to achieve peak sample efficiency[cite: 7, 12]. I used the `LunarLander-v3` environment from the Gymnasium library to implement and benchmark it[cite: 4, 12].

Concerning the environment, it simulates a rocket landing on a lunar surface with inertia, fuel constraints, and gravity dynamics.

**Objective** : Navigate the spacecraft and perform a soft landing between the two flags on the target pad (solving threshold of +200 points).  
- **State** : 8 continuous numerical values representing x & y positions, x & y linear velocities, angle, angular velocity, and two booleans indicating ground contact for each landing leg.  
- **Actions** : 4 discrete choices (0: Do nothing, 1: Fire left orientation engine, 2: Fire main engine, 3: Fire right orientation engine).  
- **Rewards** : Positive for moving toward the landing pad and landing smoothly (+100 to +200); negative for crashing (-100) or firing engines (fuel consumption penalty).

You can track the step-by-step progression and benchmarking of this project in the [report.md](report.md) file.

## Getting Started

Below are the steps you can follow to test and run the project yourself : 

- Clone the repository :
    ```bash
    git clone [https://github.com/Anirod310/lunar-lander-rainbow-dqn.git](https://github.com/Anirod310/lunar-lander-rainbow-dqn.git)
    cd lunar-lander-rainbow-dqn
    ```

- This project relies on the Python libraries listed in the [requirements.txt](requirements.txt) file. Install all dependencies by running the following command in your environment:
    ```bash
    pip install -r requirements.txt
    ```

### Running the project

- Train and evaluate the agent:
```bash
python train.py
python evaluate.py
```

For each part of the project, the config.py file contains the hyperparameters used for training and evaluating the policy. Feel free to modify them and observe how they affect convergence and stability.

You can also play the game yourself using the manual_play file : 
```bash
python manual_play.py
```
(I found out that the game is super hard in 60 ticks, it's way easier in 20, so feel free to modify ```clock.tick(60)``` to ```clock.tick(20)``` for exemple)

## Project Structure

The project is organized as follows :
```bash
lunar-lander-rainbow-dqn/
├── models/             
│   └── best_weights.pth
├── agent.py           
├── replay_buffer.py   
├── manual_play.py     
├── config.py          
├── evaluate.py         
├── model.py            
├── train.py            
├── README.md           
├── report.md           
└── requirements.txt
```
## Key Concepts learned 

Through this project, I gained hands-on experience with:

- **Rainbow DQN Integration:** Combining 6 major Deep RL extensions from scratch into a unified architecture (Double Q-Learning, Dueling Networks, Prioritized Replay, Multi-Step Learning, Noisy Networks, and Categorical DQN / C51).
- **Distributional Reinforcement Learning (C51):** Replacing scalar Q-values with 51-atom categorical probability distributions and implementing the distributional Bellman projection operator ($\Phi$)(it was a real pain..).
- **Prioritized Experience Replay (PER) and SumTree:** Building an exact binary `SumTree` data structure from scratch for PER
- **Parametric Exploration (Noisy Nets):** Implementing custom `NoisyLinear` layers with factorized Gaussian noise to replace heuristic $\epsilon$-greedy exploration with learned state-dependent curiosity.
- **Sample Efficiency Benchmarking:** Demonstrating a **+68.3 % acceleration in sample efficiency** (solving the environment in **21,792 timesteps** compared to 68,705 for baseline DDQN).

## Limitations, Possible Improvements & Next Steps

While Rainbow DQN represents the pinnacle of discrete value-based reinforcement learning, this implementation highlights specific computational characteristics and clear future directions:

### Algorithmic Limitations
- **Discrete Action Space Constraint:** Rainbow DQN only supports discrete action selection (on/off thrusters), preventing continuous control of engine throttling.
- **Computational Overhead per Step:** The C51 categorical projection loop and binary SumTree priority updates introduce higher per-step CPU execution overhead compared to basic scalar Q-learning(almost 5 times longer to finish training).

---

### Next Steps

1. **Modular RL Experimentation Framework:** Refactor the codebase into a decoupled library to enable and disable different parts to compare results and try different configurations, and then use it on different games as well such as Atari games.
2. **Continuous Action Spaces:** Transition from value-based methods to Policy Gradient and Actor-Critic algorithms (**PPO**, **SAC**) to master continuous control benchmarks (e.g., `BipedalWalker`, robotics simulation in Isaac Sim).

## Contact

If you have any questions, suggestions, or feedback about this project, feel free to reach out:

- GitHub: https://github.com/Anirod310
- Email: bousek.dorian@gmail.com

I’m always open to discussing reinforcement learning, machine learning, or related topics.
