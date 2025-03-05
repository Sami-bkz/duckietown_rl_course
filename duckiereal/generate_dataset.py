from datetime import datetime
import pathlib
import pyarrow
import pandas
import pygame
import sys
from environments.real_world_environment import DuckieBotDiscrete

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('Arrow Keys to Gym Actions')

env = DuckieBotDiscrete(robot_name=os.environ['ROBOT_NAME'])
obs, info = env.reset()

# Define the mapping from arrow keys to actions
arrow_key_to_action = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3
}


def save_dataset(dataset):
    timestamp = datetime.now().astimezone().strftime(
        "%d-%m-%Y_%Hh%Mm%Ss")  # Get current date and time in the desired format
    current_dir = pathlib.Path(__file__).parent.absolute()
    filename = f"{current_dir}/duckiebot_interactions_{timestamp}.parquet"  # Convert the date-time into a filename
    pandas.DataFrame(dataset).to_parquet(filename, engine="pyarrow", index=False)

def main():
    clock = pygame.time.Clock()

    # Reset the environment
    last_observation, _ = env.reset()
    data = []  # To store interaction samples

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_dataset(data)
                pygame.quit()
                sys.exit()

            # Check for key presses
            if event.type == pygame.KEYDOWN:
                if event.key in arrow_key_to_action:
                    action = arrow_key_to_action[event.key]
                    action_to_save = int(action)
                    observation, reward, done, terminated, truncated = env.step(action)

                    print("data build")
                    # Store interaction sample
                    data.append({
                        "s": last_observation.flatten().tolist(),
                        "a": action_to_save,
                        "r": reward,
                        "d": terminated or truncated,
                        "next_s": observation.flatten().tolist()
                    })

                    print(f"Action: {action}, Reward: {reward}, Done: {done}")

        # Cap the frame rate
        clock.tick(30)

if __name__ == "__main__":
    main()
