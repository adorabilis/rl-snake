# Reinforcement Learning Snake Game

This project implements a Deep Q-Learning (DQL) agent that learns to play the classic Snake game. The implementation uses PyTorch for the neural network and Pygame for the game environment.

## How It Works

<img align="right" src="https://github.com/user-attachments/assets/2f912ded-5699-4316-a591-b6e4ef65440a" alt="RL agent playing the classic snake game">

The agent uses a neural network to approximate the Q-value function, which predidcts the expected future rewards for each possible action. The agent explores the environment using an epsilon-greedy policy, balancing exploration (random actions) and exploitation (choosing the best-known action). The experience replay buffer stores past interactions, allowing the agent to learn from a diverse set of states and actions.

After training, the agent learns to:

- Avoid collisions with walls and its own body
- Efficiently navigate towards the food to maximise the score
- Achieve above-average scores consistently

## Setup

```shell
git clone https://github.com/adorabilis/snake-rl.git && cd snake-rl

# Train the agent
python training.py

# Test the agent
python test_agent.py
```

## Project Structure

```
snake_rl/
├── game/
│   ├── __init__.py
│   ├── snake_game.py
│   └── constants.py
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py
│   └── model.py
├── training.py
└── test_agent.py
```

## Implementation Details

### Deep Q-Network Architecture

```
Input(11) → FC(256) → ReLU → FC(256) → ReLU → FC(3)
```

### Experience Replay
The agent stores transitions $(s_t, a_t, r_t, s_{t+1})$ in a replay buffer and samples random batches for training. This breaks correlation between consecutive samples and improves learning stability.

### Training Process

The training loop follows these steps:

1. **State Observation**: Get current state $s_t$
2. **Action Selection**: Choose action $a_t$ using ε-greedy policy,

$$a_t = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s_t,a) & \text{otherwise}
\end{cases}$$

3. **Environment Step**: Execute action, receive reward $r_t$ and next state $s_{t+1}$
4. **Memory Storage**: Store transition in replay buffer
5. **Network Update**: Sample batch and update Q-network weights

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
2. Van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI, 16(2), 2094-2100.
