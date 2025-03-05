# Duckiebot in Simulation


## Codebase overview:
the duckiesim folder contains two folders: manual and rl. 
The manual folder contains files for interacting with the environment via the keyboard and recording expert datasets. 
The file [record_discrete_pygame.py](./manual/record_discrete_pygame.py) lets you record a dataset of the trajectories you play.

The rl folder contains the files needed to train a policy via reinforcement learning. 
For the RL algorithm we'll be using [Munchausen](https://arxiv.org/abs/2007.14430): 
* [munchausen.py](./rl/munchausen.py): the main file containing the Munchausen algorithm.
* [eval.py](./rl/eval.py): the file to evaluate the model obtained after training.
* [custom_reward_function.py](./rl/custom_reward_function.py): the file containing the custom reward function that you need to modify.
* [process_data_with_reward.py](./rl/process_data_with_reward.py): the file to process the data and apply the reward function.

The last file aims to process the data and apply the reward function to some dataset you have recorded.



## Reinforcement Learning:
Now it's time to train the Duckiebot to drive autonomously! We will use reinforcement learning to train the Duckiebot to follow the line. 

### Environment presentation:

#### Overview
The implemented RL environment provides an interface for training a robotic agent in a simulated framework.

#### State (Observation)
- **Format**: Sequence of 3 RGB images (image memory)
- **Dimensions**: $3 \times 3 \times 32 \times 32$ (3 RGB channels × 3 temporal images × 32×32 pixels)
- **Representation**: Tensor with dimensions $(9 \times 32 \times 32)$

Using a sequence of images as input allows the agent to perceive the environment's dynamics and make informed decisions based on the history of observations.

#### Action Space
- **Discrete set**: $\mathcal{A}=\{0,...,9\}$
- **Meaning**: Each action corresponds to an index in a discretization of two parameters:
  - Robot's absolute speed (linear velocity)
  - Robot's angular velocity (rotation)
- **Configuration**: The discretization is defined in [discrete_env.py](../duckietownrl/gym_duckietown/envs/duckietown_discrete_env.py)

#### Reward Function
- The reward function must be implemented by the user
- File to modify: [custom_reward_function](./rl/custom_reward_function.py)
- This function determines the reward signal the agent will receive after each action

#### Important note
To adapt the environment to your specific task, configure the reward function according to the behaviors you wish to encourage or discourage.

### Reward function :
You need to define the reward function in the file [custom_reward_function.py](./rl/custom_reward_function.py). To do this, we've used standard image analysis (available in the file) to extract certain variables from the images that you can use for your reward function. 

### Parameter tuning:
The main hp to tune in the Munchausen algorithm are:
- $\alpha$
- $\tau$
In Reinforcement Leanrning in general, the temperature related to entropy regularization is often 0.1 and the temperature related to kl regularization can be set to 0.2.
So which values would you choose for $\alpha$ and $\tau$?