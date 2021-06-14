import numpy as np
import pygame

from .constants import *


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)
        self.head = [WIDTH//2, HEIGHT//2]
        self.snake = [self.head.copy()]
        self.food = self._place_food()
        self.score = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = np.random.randint(1, (WIDTH//GRID_SIZE)-1) * GRID_SIZE
            y = np.random.randint(1, (HEIGHT//GRID_SIZE)-1) * GRID_SIZE
            if [x, y] not in self.snake:
                return [x, y]

    def _get_state(self):
        # State: danger [left, straight, right], direction [4], food position [2]
        state = [
            # Danger directions
            self._is_collision((self.direction[1], -self.direction[0])),  # Left
            self._is_collision(self.direction),  # Straight
            self._is_collision((-self.direction[1], self.direction[0])),  # Right

            # Direction
            self.direction[0] == 1,  # Right
            self.direction[0] == -1, # Left
            self.direction[1] == 1,  # Down
            self.direction[1] == -1, # Up

            # Food location
            self.food[0] < self.head[0],  # Food left
            self.food[0] > self.head[0],  # Food right
            self.food[1] < self.head[1],  # Food up
            self.food[1] > self.head[1]   # Food down
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, direction):
        next_head = [self.head[0] + direction[0]*GRID_SIZE,
                     self.head[1] + direction[1]*GRID_SIZE]
        return (next_head[0] < 0 or next_head[0] >= WIDTH or
                next_head[1] < 0 or next_head[1] >= HEIGHT or
                next_head in self.snake)

    def step(self, action):
        # Action: [straight, right, left]
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right
            new_dir = clock_wise[(idx+1)%4]
        else:  # Left
            new_dir = clock_wise[(idx-1)%4]

        self.direction = new_dir
        self.head[0] += self.direction[0] * GRID_SIZE
        self.head[1] += self.direction[1] * GRID_SIZE

        done = False
        reward = 0

        if self._is_collision((0,0)) or self.head in self.snake[:-1]:
            done = True
            reward = -10
            return reward, done, self.score

        self.snake.insert(0, self.head.copy())

        if self.head == self.food:
            reward = 20
            self.score += 1
            self.food = self._place_food()
        else:
            reward = -0.2
            self.snake.pop()

        return reward, done, self.score

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill(BLACK)
        for pos in self.snake:
            pygame.draw.rect(self.screen, GREEN, (pos[0], pos[1], GRID_SIZE-1, GRID_SIZE-1))
        pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], GRID_SIZE-1, GRID_SIZE-1))
        pygame.display.flip()
        self.clock.tick(SPEED)
