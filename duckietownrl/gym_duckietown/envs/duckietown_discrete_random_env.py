# coding=utf-8
import numpy as np
from gymnasium import spaces
from duckiesim.rl.custom_reward_function import compute_custom_reward
from ..simulator import Simulator
from .. import logger
import cv2
import numpy as np



def random_brightness(image, delta=50):
    """Ajuste aléatoirement la luminosité"""
    beta = np.random.uniform(-delta, delta)
    return np.clip(image + beta, 0, 255).astype(np.uint8)

def random_contrast(image, lower=0.5, upper=1.5):
    """Ajuste aléatoirement le contraste"""
    alpha = np.random.uniform(lower, upper)
    return np.clip(alpha * image, 0, 255).astype(np.uint8)

def random_noise(image, noise_type="gaussian"):
    """Ajoute du bruit aléatoire"""
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.004
        noisy = np.copy(image)
        # Salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                 for i in image.shape]
        noisy[coords] = 255
        # Pepper
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                 for i in image.shape]
        noisy[coords] = 0
        return noisy

def random_blur(image, kernel_range=(1, 5)):
    """Applique un flou aléatoire"""
    kernel_size = np.random.randint(kernel_range[0], kernel_range[1]) * 2 + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def random_rotation(image, angle_range=(-10, 10)):
    """Rotation aléatoire"""
    angle = np.random.uniform(angle_range[0], angle_range[1])
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

def random_color_shift(image, intensity=0.5):
    """Modifie aléatoirement les canaux de couleur"""
    shifted = image.astype(np.float32)
    for i in range(3):  # Pour chaque canal RGB
        shift = np.random.uniform(-intensity, intensity)
        shifted[:,:,i] = np.clip(shifted[:,:,i] * (1 + shift), 0, 255)
    return shifted.astype(np.uint8)

def random_perspective(image, intensity=0.05):
    """Applique une déformation perspective aléatoire"""
    height, width = image.shape[:2]
    
    # Points de référence
    pts1 = np.float32([[0,0], [width,0], [0,height], [width,height]])
    
    # Points déformés aléatoirement
    pts2 = np.float32([[0+np.random.randint(-intensity*width, intensity*width),
                        0+np.random.randint(-intensity*height, intensity*height)],
                       [width+np.random.randint(-intensity*width, intensity*width),
                        0+np.random.randint(-intensity*height, intensity*height)],
                       [0+np.random.randint(-intensity*width, intensity*width),
                        height+np.random.randint(-intensity*height, intensity*height)],
                       [width+np.random.randint(-intensity*width, intensity*width),
                        height+np.random.randint(-intensity*height, intensity*height)]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (width, height))

# Exemple d'utilisation combinée
def apply_random_augmentation(image):
    """Applique aléatoirement plusieurs augmentations"""
    augmentations = [
        random_brightness,
        random_contrast,
        random_noise,
        random_blur,
        random_rotation,
        random_color_shift,
        random_perspective
    ]
    
    # Applique aléatoirement une augmentation
    augmentation = np.random.choice(augmentations)
    return augmentation(image)


class DuckietownDiscretRandomEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, 
                 gain=1.0, 
                 trim=0.0, 
                 radius=0.0318, 
                 k=27.0, 
                 limit=1.0, 
                 discretization_step=3, 
                 activate_action_noise= True, 
                 activate_parameter_noise= True, 
                 noise_parameter_amplitute = 0.1,
                 nb_frames = 3, 
                 vmax = 0.5, 
                 rad_max = np.pi/2,   
                 **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")
        # action space
        dim_action_space = 2
        self.action_space = spaces.Discrete(discretization_step**dim_action_space)
        self.vmax = vmax
        self.rad_max = rad_max
        self.actions = [(x,y) for x in np.linspace(-self.vmax, self.vmax, discretization_step).tolist() for y in np.linspace(-self.rad_max, self.rad_max, discretization_step).tolist()]
        self.action_noise_std = 2/((discretization_step-1)*2) # should add 2.33 to unsure that the noise respect Bienaymé-Tchebychev inequality
        self.activate_noise = activate_action_noise    
        
        # observation space
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera_width, self.camera_height, 3*nb_frames), dtype=np.uint8)
        self.nb_frames = nb_frames
        self.obs = np.zeros((self.camera_width, self.camera_height, 3*nb_frames), dtype=np.uint8)
        
        
        # ENV parameters
        self.activate_parameter_noise = activate_parameter_noise
        self.noise_parameter_amplitute = noise_parameter_amplitute
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
        
        # noise the parameters of the environment
        gain = self.gain + np.random.normal(0, self.noise_parameter_amplitute * np.abs(self.gain))
        trim = self.trim + np.random.normal(0, self.noise_parameter_amplitute * np.abs(self.trim))
        radius = self.radius + np.random.normal(0, self.noise_parameter_amplitute * np.abs(self.radius))
        k = self.k + np.random.normal(0, self.noise_parameter_amplitute * np.abs(self.k))
        
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
        k_r = k
        k_l = k

        # adjusting k by gain and trim
        k_r_inv = (gain + trim) / k_r
        k_l_inv = (gain - trim) / k_l

        omega_r = (vel + 0.5 * dangle * baseline) / radius
        omega_l = (vel - 0.5 * dangle * baseline) / radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        
        # obs_plot_normal = cv2.resize(obs, (256,256), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("original", obs_plot_normal)
        
        # # check random brightness
        # obs_random_brightness = random_brightness(obs_plot_normal)
        # cv2.imshow("random_brightness", obs_random_brightness)

        # # check random contrast
        # obs_random_contrast = random_contrast(obs_plot_normal)
        # cv2.imshow("random_contrast", obs_random_contrast)

        # # check random noise
        # obs_random_noise = random_noise(obs_plot_normal)
        # cv2.imshow("random_noise", obs_random_noise)

        # # check random blur
        # obs_random_blur = random_blur(obs_plot_normal)
        # cv2.imshow("random_blur", obs_random_blur)

        # # check random rotation
        # obs_random_rotation = random_rotation(obs_plot_normal)
        # cv2.imshow("random_rotation", obs_random_rotation)

        # # check random color shift
        # obs_random_color_shift = random_color_shift(obs_plot_normal)
        # cv2.imshow("random_color_shift", obs_random_color_shift)

        # # check random perspective
        # obs_random_perspective = random_perspective(obs_plot_normal)
        # cv2.imshow("random_perspective", obs_random_perspective)


        
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


class DuckietownLF(DuckietownDiscretRandomEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownDiscretRandomEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownDiscretRandomEnv.step(self, action)
        return obs, reward, done, info
