from datetime import datetime
import pathlib
import pyarrow
import pandas
import pygame
import os
import sys
from environments.real_world_environment import DuckieBotDiscrete

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('Arrow Keys to Gym Actions')

env = DuckieBotDiscrete(robot_name=os.environ['ROBOT_NAME'])

# Define the mapping from arrow keys to actions
arrow_key_to_action = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3
}


def main():
    clock = pygame.time.Clock()

    # Reset the environment
    env.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Check for key presses
            if event.type == pygame.KEYDOWN:
                if event.key in arrow_key_to_action:
                    action = arrow_key_to_action[event.key]
                    observation, reward, done, terminated, truncated = env.step(action)
                    print(f"Action: {action}, Reward: {reward}, Done: {done}")

        # Cap the frame rate
        clock.tick(30)

if __name__ == "__main__":
    main()
