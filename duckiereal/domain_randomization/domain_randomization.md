# Domain Randomization

## Formal Definition

Domain Randomization is a technique that involves systematically randomizing the parameters of the environment's transition kernel during training. In reinforcement learning, if we define a Markov Decision Process (MDP) as a tuple (S, A, P, R, γ) where P represents the transition probability function, domain randomization works by introducing variations to P during training. This creates a distribution of environments with different dynamics, forcing the agent to learn policies that are robust across this distribution rather than optimized for a single environment configuration.

The underlying principle is that by training on a sufficiently diverse set of simulated environments, the resulting policy will be robust enough to work in the real environment, even if the simulation is not a perfect match for reality (bridging the "reality gap").

## Implementation in Our Project

In our implementation for a car racing environment with image inputs, we applied domain randomization to both the environment dynamics and visual aspects. These implementations can be found in the file: [duckietown_discrete_random_env.py](../../duckietownrl/gym_duckietown/envs/duckietown_discrete_random_env.py).

## Types of Implemented Randomization

1. **Environment Dynamics Randomization (Domain Randomization)**
   - Motor gain noise
   - Steering trim noise
   - Wheel radius noise
   - Motor constant noise
   - Action perturbations (speed and steering angle)

2. **Visual Input Randomization (Data Augmentation)**
   - Random brightness adjustment
   - Random contrast variation
   - Random noise (Gaussian and salt-and-pepper)
   - Random Gaussian blur
   - Random rotation
   - Color channel modification
   - Perspective distortion

## Best Practices

1. **Progressive Application**
   - Start with mild randomization
   - Gradually increase intensity
   - Monitor impact on performance

2. **Balance**
   - Maintain a proportion of non-randomized environments
   - Avoid extreme randomization values
   - Keep transformations realistic

3. **Monitoring**
   - Track performance in the real environment
   - Adjust probabilities and intensities based on results
   - Visually verify the range of randomization

## Advantages

- Improves model robustness
- Reduces overfitting to specific environment configurations
- Aids in generalization to unseen scenarios
- Simulates varied conditions without requiring explicit modeling
- Helps bridge the sim-to-real gap

## Limitations

- Can slow down learning
- Requires fine-tuning of randomization parameters
- May degrade performance if poorly calibrated
- Cannot fully account for systematic discrepancies between simulation and reality

This approach enables training more robust agents capable of generalizing to different visual and physical conditions they might encounter in real-world situations, ultimately helping to address the sim-to-real transfer problem in robotics and reinforcement learning.