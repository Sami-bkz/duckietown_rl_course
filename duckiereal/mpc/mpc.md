# Model Predictive Control and Latent Space Planning

## Core Intuition

In many tasks, learning an optimal policy directly can be challenging due to:
- Complex action-state relationships
- Long-term dependencies
- High-dimensional state spaces
- Sparse rewards

However, learning the environment dynamics $(s_{t+1} = f(s_t, a_t))$ is often more straightforward because:
- It's a supervised learning problem
- Each state transition provides immediate feedback
- The relationship is more consistent
- It doesn't depend on long-term rewards

This leads to a two-phase approach:
1. First learn how the world works (dynamics model)
2. Then use this model to plan and improve policy

Think of it like:
- A chess player first learns how pieces move
- Then uses this knowledge to plan strategies

```python
# Simplified concept
class ModelBasedRL:
    def __init__(self):
        self.dynamics_model = DynamicsModel()
        self.policy = Policy()
        
    def train(self):
        # Phase 1: Learn dynamics
        self.dynamics_model.train(experience_data)
        
        # Phase 2: Improve policy using model
        while not converged:
            trajectories = self.plan_with_model()
            self.policy.improve(trajectories)
```


## Model Predictive Control (MPC) in RL Context

Model Predictive Control is a control strategy that:
1. Uses a model to predict future states
2. Optimizes actions over a finite horizon
3. Executes the first action
4. Repeats the process (rolling horizon)

In RL terms:
- The model predicts $s_{t+1} = f(s_t, a_t)$
- We search for optimal action sequence $a_{t:t+H}$ that maximizes expected reward
- Similar to value iteration but over a finite horizon

```python
def mpc_step(current_state, model, horizon=10):
    best_sequence = None
    best_value = float('-inf')
    
    # Search for best action sequence
    action_sequences = generate_action_sequences()
    for sequence in action_sequences:
        value = 0
        state = current_state
        
        # Simulate trajectory
        for action in sequence:
            next_state = model.predict(state, action)
            reward = get_reward(state, action, next_state)
            value += reward
            state = next_state
            
        if value > best_value:
            best_value = value
            best_sequence = sequence
            
    return best_sequence[0]  # Return first action
```

## Latent Space Planning

Instead of planning in high-dimensional state space (like images), we:
1. Learn a low-dimensional latent representation
2. Plan in this compact space
3. Decode actions back to original space

### Architecture:
```
High-dim State → Encoder → Latent State → Planning → Actions
                                  ↑
                            Latent Dynamics
```

```python
class LatentPlanner:
    def __init__(self):
        self.encoder = Encoder()  # State → Latent
        self.decoder = Decoder()  # Latent → State
        self.dynamics = LatentDynamics()  # Predicts next latent state
        
    def plan(self, state):
        z = self.encoder(state)
        z_sequence = self.plan_latent(z)
        return self.decode_actions(z_sequence)
```

## MCTS for Planning

Monte Carlo Tree Search is particularly suitable because:
- Can handle large action spaces
- Balances exploration/exploitation
- Works well with learned models

Basic MCTS structure:
```python
class MCTSNode:
    def __init__(self, state, model):
        self.state = state
        self.model = model
        self.children = {}
        self.visits = 0
        self.value = 0
        
    def select(self):
        # UCB1 selection
        return max(self.children.items(), 
                  key=lambda x: x[1].ucb_score())
        
    def expand(self):
        # Add new child nodes
        
    def simulate(self):
        # Rollout policy
        
    def backpropagate(self, value):
        # Update statistics
```

## Complete Pipeline

1. **Encoding**:
   - Convert high-dim state to latent representation
   - Learn compact dynamics model

2. **Planning**:
   - Use MCTS in latent space
   - Evaluate trajectories using learned model
   - Balance exploration/exploitation

3. **Execution**:
   - Select best action
   - Apply to environment
   - Repeat process

```python
class LatentMPCController:
    def __init__(self):
        self.encoder = Encoder()
        self.dynamics = LatentDynamics()
        self.mcts = MCTSPlanner()
        
    def act(self, state):
        z = self.encoder(state)
        plan = self.mcts.plan(
            initial_state=z,
            dynamics_model=self.dynamics,
            horizon=10
        )
        return plan.get_action()
```

## Advantages

- **Efficiency**: Planning in lower-dimensional space
- **Scalability**: Can handle complex environments
- **Sample Efficiency**: Uses learned model
- **Adaptability**: Replanning at each step

## Limitations

- Quality depends on learned representations
- Computational cost of planning
- Model errors can compound
- Requires good latent space structure

This approach combines the benefits of model-based planning with efficient latent representations, making it suitable for complex visual control tasks.