from config import CONFIG
from agent import Agent
import gymnasium as gym
import torch
import random
import numpy as np

def set_seed(env, seed=42):
    random.seed(seed)

    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    env.action_space.seed(seed)

env = gym.make("LunarLander-v3")

set_seed(env, seed=42)

agent = Agent(CONFIG["state_dim"], CONFIG["action_dim"],config=CONFIG)

scores = []

best_score = 315

total_timesteps = 0

state, info = env.reset(seed=42)

for episode in range(CONFIG["num_episodes"]):
    score_episode = 0

    if episode != 0:
        state,info = env.reset()

    done = False

    while not done :

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        total_timesteps += 1

        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)

        state = next_state

        score_episode += reward

    agent.epsilon = max(agent.epsilon*CONFIG["epsilon_decay"], CONFIG["epsilon_end"])

    scores.append(score_episode)

    if episode % 100 == 0 :
        print(f"Episode {episode} | Score : {score_episode} | Epsilon : {agent.epsilon}")

    if score_episode > 200:
        print(f"Perfect landing ! number of timesteps : {total_timesteps}")

    if score_episode > best_score : 
        print(f"Perfect landing higher than the best score ! Score {score_episode} | Saving weights...")
        best_score = score_episode
        torch.save(agent.q_network.state_dict(), f"models/best_weights.pth")