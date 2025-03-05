#!/usr/bin/env python
# manual_control_pynput

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows with pynput.
"""
from PIL import Image
import argparse
import sys
import gym
import numpy as np
from pynput import keyboard
import cv2
from duckietownrl.gym_duckietown.envs import DuckietownEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown")
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

# Initialize the environment
if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
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
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

# Initialize action and control states
action = np.array([0.0, 0.0])
key_states = {}

def on_press(key):
    global action, key_states

    try:
        if key.char == "r":  # Reset
            print("RESET")
            env.reset()
            env.render()
        elif key.char == "\r":  # Save screenshot
            print("Saving screenshot")
            img = env.render("rgb_array")
            im = Image.fromarray(img)
            im.save("screenshot.png")
    except AttributeError:
        pass

    if key == keyboard.Key.esc:  # Exit
        env.close()
        sys.exit(0)
    elif key == keyboard.Key.up:
        key_states["up"] = True
    elif key == keyboard.Key.down:
        key_states["down"] = True
    elif key == keyboard.Key.left:
        key_states["left"] = True
    elif key == keyboard.Key.right:
        key_states["right"] = True
    elif key == keyboard.Key.space:
        key_states["space"] = True
    elif key == keyboard.Key.shift:
        key_states["shift"] = True

def on_release(key):
    global key_states

    if key == keyboard.Key.up:
        key_states["up"] = False
    elif key == keyboard.Key.down:
        key_states["down"] = False
    elif key == keyboard.Key.left:
        key_states["left"] = False
    elif key == keyboard.Key.right:
        key_states["right"] = False
    elif key == keyboard.Key.space:
        key_states["space"] = False
    elif key == keyboard.Key.shift:
        key_states["shift"] = False

def update():
    global action

    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_states.get("up", False):
        action += np.array([0.44, 0.0])
    if key_states.get("down", False):
        action -= np.array([0.44, 0])
    if key_states.get("left", False):
        action += np.array([0, 1])
    if key_states.get("right", False):
        action -= np.array([0, 1])
    if key_states.get("space", False):
        action = np.array([0, 0])

    # Adjust curvature
    v1, v2 = action[0], action[1]
    # if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
    #     delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
    #     v1 += delta_v
    #     v2 -= delta_v

    action[0], action[1] = v1, v2

    # Speed boost
    if key_states.get("shift", False):
        action *= 1.5

    obs, reward, done, _ = env.step(action)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    # print("action = %s" % action)

    if done:
        print("done!")
        env.reset()
        env.render()

    img = env.render()
    cv2.imshow("image", img)
    cv2.waitKey(1)

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Main loop
try:
    while True:
        update()
except KeyboardInterrupt:
    env.close()
