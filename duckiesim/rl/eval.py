import torch
import gymnasium as gym
import numpy as np
from munchausen import QNetwork
import plotext as plt
import cv2

def evaluate_with_rendering(model_path, env_id, num_episodes=10, seed=0, tau_soft=1.0):
    """
    Évalue le modèle sauvegardé avec rendu visuel, en utilisant une politique softmax.

    Args:
        model_path (str): Le chemin vers le fichier du modèle sauvegardé.
        env_id (str): L'identifiant de l'environnement.
        num_episodes (int): Le nombre d'épisodes à évaluer.
        seed (int): La graine initiale pour reproduire les résultats.
        tau_soft (float): Température pour le softmax.
    """
    # Charger le modèle
    env = gym.make(env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(env.action_space, env.observation_space).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Évaluation
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        # obs, _ = env.reset(seed=58)

        done = False
        total_reward = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = model(obs_tensor)
                policy = torch.softmax(q_values / tau_soft, dim=-1)
                # print('policy', policy)
                # argmax
                action = torch.argmax(policy, dim=-1).item() 
                # action = torch.multinomial(policy, num_samples=1).item()
                # print('action', action)
            img = env.render()
            cv2.imshow("Duckietown", img)
            cv2.waitKey(1)
            obs, reward, done, _, _ = env.step(action)
            # print('obs', obs.shape)
            total_reward += reward
            print(f"step {env.unwrapped.episodic_length} reward {reward}")

        print(f"Récompense totale de l'épisode : {total_reward}")

    env.close()

if __name__ == "__main__":
    model_path = "path_to_your_model"  # Adapter au chemin réel
    env_id = "DuckietownDiscrete-v0"

    evaluate_with_rendering(model_path, env_id, num_episodes=20, seed=42, tau_soft=0.05)