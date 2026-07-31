import gymnasium as gym
import torch
from agent import Agent
from config import CONFIG
import pygame
import sys

env = gym.make("LunarLander-v3", render_mode="human")

agent = Agent(CONFIG["state_dim"], CONFIG["action_dim"], config=CONFIG)

model = agent.q_network.load_state_dict(torch.load("models/best_weights.pth", weights_only=True))

agent.epsilon = 0.0

for episode in range(CONFIG["num_test_episodes"]):
    score_episode = 0

    state, info = env.reset()

    done = False

    while not done :

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                print("Fermeture manuelle de la fenêtre.")
                env.close()
                sys.exit()

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        state = next_state

        score_episode += reward

    print(f"Episode : {episode} | Score : {score_episode}")

env.close()
