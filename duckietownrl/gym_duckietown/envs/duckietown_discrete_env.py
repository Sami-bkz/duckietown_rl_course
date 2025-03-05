# coding=utf-8
import numpy as np
from gymnasium import spaces
from duckiesim.rl.custom_reward_function import compute_custom_reward
from ..simulator import Simulator
from .. import logger


class DuckietownDiscretEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, discretization_step=3, activate_noise= False, nb_frames = 3, vmax = 0.5, rad_max = np.pi/2,   **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")
        # action space
        dim_action_space = 2
        self.action_space = spaces.Discrete(discretization_step**dim_action_space)
        self.vmax = vmax
        self.rad_max = rad_max
        self.actions = [(x,y) for x in np.linspace(-self.vmax, self.vmax, discretization_step).tolist() for y in np.linspace(-self.rad_max, self.rad_max, discretization_step).tolist()]
        self.action_noise_std = 2/((discretization_step-1)*2) # should add 2.33 to unsure that the noise respect Bienaymé-Tchebychev inequality
        self.activate_noise = activate_noise    
        
        # observation space
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera_width, self.camera_height, 3*nb_frames), dtype=np.uint8)
        self.nb_frames = nb_frames
        self.obs = np.zeros((self.camera_width, self.camera_height, 3*nb_frames), dtype=np.uint8)
        
        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit
        
        # episodic return 
        self.episodic_return = 0
        # episode length
        self.episodic_length = 0
        
    def reset(self, seed = np.random.randint(0,1000), options = None):
        self.seed(seed)
        obs = Simulator.reset(self)
        # repeat the same frame for nb_frames to get (camera_width, camera_height, 3*nb_frames) shape
        self.obs = np.repeat(obs, self.nb_frames, axis=2)
        self.episodic_return = 0
        self.episodic_length = 0
        return self.obs.copy(), {}
    
    def step(self, action_idx):
        vel, dangle = self.actions[action_idx]
        
        # add noise
        vel += np.random.normal(0, self.action_noise_std) if self.activate_noise else 0
        # clip 
        vel = np.clip(vel, -self.vmax, self.vmax)
        
        dangle += np.random.normal(0, self.action_noise_std) if self.activate_noise else 0
        # clip
        dangle = np.clip(dangle, -self.rad_max, self.rad_max)

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * dangle * baseline) / self.radius
        omega_l = (vel - 0.5 * dangle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        
        # custom reward
        reward = compute_custom_reward(obs, self.actions[action_idx]) if not done else reward
        
        # update the observation in FIFO order
        self.obs = np.concatenate([self.obs[:,:,3:], obs], axis=2)
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine
        self.episodic_return += reward
        info["episodic_return"] = self.episodic_return
        self.episodic_length += 1
        info["episodic_length"] = self.episodic_length
        # info["obs"] = obs
        return self.obs.copy(), reward, done, False, info


class DuckietownLF(DuckietownDiscretEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownDiscretEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownDiscretEnv.step(self, action)
        return obs, reward, done, info
