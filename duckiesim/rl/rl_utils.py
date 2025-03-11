import gymnasium as gym
import torch
from torch.nn import functional as F
import numpy as np



def evaluate(agent, env_id, num_episodes=10, seed=0, tau = 0.03):
    """
    Évalue l'agent sur plusieurs trajectoires avec des graines différentes.

    Args:
        agent (QNetwork): Le réseau de neurones entraîné.
        env_id (str): L'identifiant de l'environnement.
        num_episodes (int): Le nombre d'épisodes pour l'évaluation.
        seed (int): La graine initiale pour reproduire les résultats.

    Returns:
        float: Le score moyen sur les épisodes d'évaluation.
    """
    device = next(agent.parameters()).device
    env = gym.make(env_id)
    scores = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent(obs_tensor)
                policy = F.softmax(q_values / tau, dim=-1)
                action = np.array(torch.multinomial(policy, 1).squeeze(-1).cpu().tolist(), dtype=np.int32)[0]
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        scores.append(total_reward)

    env.close()
    return np.mean(scores)



def evaluate_ppo(agent, env_id, num_episodes=10, seed=0):
    """
    Évalue l'agent sur plusieurs trajectoires avec des graines différentes.

    Args:
        agent (QNetwork): Le réseau de neurones entraîné.
        env_id (str): L'identifiant de l'environnement.
        num_episodes (int): Le nombre d'épisodes pour l'évaluation.
        seed (int): La graine initiale pour reproduire les résultats.

    Returns:
        float: Le score moyen sur les épisodes d'évaluation.
    """
    device = next(agent.parameters()).device
    env = gym.make(env_id)
    scores = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _,_,_= agent.get_action_and_value(obs_tensor)
                action = np.array(action.cpu().tolist(), dtype=np.int32)[0]
                obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        scores.append(total_reward)

    env.close()
    return np.mean(scores)