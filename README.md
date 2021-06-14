# Reinforcement Learning Snake Game

This project implements a Deep Q-Learning (DQL) agent that learns to play the classic Snake game. The implementation uses PyTorch for the neural network and Pygame for the game environment.

## How It Works

The agent uses a neural network to approximate the Q-value function, which predidcts the expected future rewards for each possible actions. The agent explores the environment using an epsilon-greedy policy, balancing exploration (random actions) and exploitation (choosing the best-known action). The experience replay buffer stores past interactions, allowing the agent to learn from a diverse set of states and actions.

After training, the agent learns to:

- Avoid collisions with walls and its own body
- Efficiently navigate towards the food to maximise the score.
- Achieve above-average scores consistently.

## Setup

```shell
git clone https://github.com/adorabilis/snake-rl.git && cd snake-rl

# Train the agent
python training.py

# Test the agent
python test_agent.py
```
