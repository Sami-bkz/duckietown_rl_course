#!/usr/bin/env python
# manual_control_pygame

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows with Pygame.
"""

from PIL import Image
import argparse
import sys
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import pygame
import cv2
from duckietownrl.gym_duckietown.envs import DuckietownDiscretEnv
import time

# --- Arguments de la ligne de commande ---
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown")
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--data-save", default=False, action="store_true", help="save expert data")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()


# --- Initialisation de pygame ---
pygame.init()
# Créer une fenêtre visible, mais très petite pour la gestion des touches
window = pygame.display.set_mode((1, 1))
pygame.display.set_caption("")  # Titre vide

# --- Initialisation de l'environnement ---
env = DuckietownDiscretEnv(
    seed=args.seed,
    map_name=args.map_name,
    draw_curve=args.draw_curve,
    draw_bbox=args.draw_bbox,
    domain_rand=args.domain_rand,
    frame_skip=args.frame_skip,
    distortion=args.distortion,
    camera_rand=args.camera_rand,
    dynamics_rand=args.dynamics_rand,
)

env.reset()
env.render()
img = env.render("rgb_array")
cv2.imshow("image", img)
cv2.waitKey(1)





print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')

# --- Initialisation des variables globales ---
action = 0
key_states = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "space": False
}

# Pour enregistrer les données
data = []

# --- Programme principal ---
if __name__ == "__main__":
    print("Écouteur de clavier actif. Utilisez les flèches pour contrôler le robot.")
    print("La fenêtre Pygame doit avoir le focus - cliquez dessus si nécessaire.")
    print("r: reset, Entrée: capture d'écran, Échap: quitter")
    
    seed = 0
    
    # Boucle principale
    try:
        total_reward = 0.0
        current_obs, _ = env.reset(seed=seed)  # Réinitialisation de l'environnement
        running = True

        while running:
            # Traitement des événements Pygame
            for event in pygame.event.get():
                # pygame.display.flip()
                if event.type == pygame.QUIT:
                    # running = False
                    pass
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset
                        print("RESET")
                        env.reset()
                        env.render()
                    elif event.key == pygame.K_RETURN:  # Save screenshot
                        print("Saving screenshot")
                        img = env.render("rgb_array")
                        Image.fromarray(img).save("screenshot.png")
                    elif event.key == pygame.K_ESCAPE:  # Exit
                        # running = False
                        pass
                    elif event.key == pygame.K_UP:
                        key_states["up"] = True
                    elif event.key == pygame.K_DOWN:
                        key_states["down"] = True
                    elif event.key == pygame.K_LEFT:
                        key_states["left"] = True
                    elif event.key == pygame.K_RIGHT:
                        key_states["right"] = True
                    elif event.key == pygame.K_SPACE:
                        key_states["space"] = True
                        
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        key_states["up"] = False
                    elif event.key == pygame.K_DOWN:
                        key_states["down"] = False
                    elif event.key == pygame.K_LEFT:
                        key_states["left"] = False
                    elif event.key == pygame.K_RIGHT:
                        key_states["right"] = False
                    elif event.key == pygame.K_SPACE:
                        key_states["space"] = False
                    print(f"key released: {pygame.key.name(event.key)}")
            
            # Détermine l'action en fonction des touches pressées
            if key_states["up"]:
                print("up") 
                action = 7  # Accélération avant (exemple)
            elif key_states["down"]:
                action = 1  # Recul ou arrêt
            elif key_states["left"]:
                action = 8  # Tourne à gauche
            elif key_states["right"]:
                action = 6  # Tourne à droite
            elif key_states["space"]:
                action = 4  # Action spéciale ou arrêt

            # Si aucune touche n'est pressée, définir l'action d'arrêt
            if not any(key_states.values()):
                action = 4  # Action immobile (arrêt complet)

            # Effectue une étape de simulation
            next_obs, reward, done, _, _ = env.step(action)
            time.sleep(0.002)
            print(f"next obs shape: {len(next_obs.flatten().tolist())}")  
            
            # Enregistre les données de la simulation
            data.append({
                "s": current_obs.flatten().tolist(),
                "a": action,
                "r": reward,
                "d": done,
                "next_s": next_obs.flatten().tolist()
            })

            total_reward += reward
            current_obs = next_obs  # Mise à jour de l'observation courante

            print(f"step_count = {env.unwrapped.step_count}, reward = {reward:.3f}, action = {action}")

            if done:
                seed += 1
                print("done!")
                print(f"Total reward: {total_reward}")
                time.sleep(2)
                current_obs, _ = env.reset(seed=seed)  # Réinitialisation de l'environnement
                total_reward = 0.0

            # Affiche l'image de l'environnement
            img = env.render("rgb_array")
            cv2.imshow("image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Garde la fenêtre Pygame active
            # pygame.display.flip()
            print("running",  running)
        print("Programme terminé.")
    except KeyboardInterrupt:
        print("Programme arrêté par l'utilisateur.")
    finally:
        print("Fermeture de l'environnement.")
        if args.data_save:
            # --- Configuration du chemin de sortie ---
            script_dir = os.path.dirname(os.path.realpath(__file__))  # Répertoire du script
            output_dir = os.path.join(script_dir, "dataset")         # Dossier 'dataset'
            output_file = os.path.join(output_dir, "expert_data"+'_'+str(len(data))+".parquet")  # Fichier de sortie

            # Crée le dossier 'dataset' s'il n'existe pas
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Le dossier '{output_dir}' a été créé.")
            # Enregistre les données au format Parquet
            df = pd.DataFrame(data)
            df.to_parquet(output_file, engine="pyarrow", index=False)
            print(f"Les données ont été sauvegardées dans le fichier : {output_file}")
        
        env.close()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit(0)
