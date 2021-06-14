import torch

from agent.dqn_agent import DQNAgent
from agent.model import DQN
from game.snake_game import SnakeGame

env = SnakeGame()
agent = DQNAgent(state_size=11, action_size=3)
agent.model.load_state_dict(torch.load("dqn_model_200.pth"))
agent.epsilon = 0.0  # Disable exploration

state = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.act(state)
    reward, done, score = env.step([1 if i == action else 0 for i in range(3)])
    total_reward += reward
    state = env._get_state()
    env.render()

print(f"Final Score: {score}")
