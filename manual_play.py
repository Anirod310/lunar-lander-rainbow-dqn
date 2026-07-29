import gymnasium as gym 
import pygame

env = gym.make("LunarLander-v3", render_mode="human")

state, info = env.reset()

clock = pygame.time.Clock()

running = True

score = 0

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = 0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        action = 3
    elif keys[pygame.K_UP]:
        action = 2
    elif keys[pygame.K_RIGHT]:
        action = 1

    next_state, reward, terminated, truncated, info = env.step(action)

    score += reward

    if terminated or truncated:
        print(f"Score : {score}")
        state, info = env.reset()
        score = 0

    clock.tick(60)

env.close()
