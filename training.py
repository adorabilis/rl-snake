import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from game.snake_game import SnakeGame

EPISODES = 500
BATCH_SIZE = 128

env = SnakeGame()
agent = DQNAgent(state_size=11, action_size=3)
for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        reward, done, score = env.step([1 if i == action else 0 for i in range(3)])
        next_state = env._get_state()
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay(BATCH_SIZE)

        env.render()
        # print(f"Action: {action}, Reward: {reward}, Done: {done}")

    print(f"Episode: {e + 1}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

    if (e + 1) % 50 == 0:
        torch.save(agent.model.state_dict(), f"dqn_model_{e + 1}.pth")
