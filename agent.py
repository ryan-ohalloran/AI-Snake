from sre_parse import State
from tkinter.tix import MAX
import torch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        # randomness
        self.epsilon = 0
        # discount rate
        self.gamma = 0
        # mem - if we exeed, automatically removes elements from the left
        self.memory = deque(maxlen=MAX_MEMORY)
        # model
        self.model = None
        self.trainer = None
        


    def get_state(self, game):
        head = game.snake[0]
        # points in all directions
        pointLeft  = Point(head.x - 20, head.y)
        pointRight = Point(head.x + 20, head.y)
        pointUp    = Point(head.x, head.y - 20)
        pointDown  = Point(head.x, head.y + 20)
        # all directions otherwise
        dirLeft = game.direction == Direction.LEFT
        dirRight = game.direction == Direction.RIGHT
        dirUp    = game.direction == Direction.UP
        dirDown = game.direction == Direction.DOWN
        # state is an array constituting current condition
        state = [
            # if there is danger straight ahead
            (dirRight and game.is_collision(pointRight)) or
            (dirLeft  and game.is_collision(pointLeft))  or
            (dirUp    and game.is_collision(pointUp))    or
            (dirDown  and game.is_collision(pointDown)),
            # if there is danger to the right
            (dirUp    and game.is_collision(pointRight)) or
            (dirDown  and game.is_collision(pointLeft))  or
            (dirLeft  and game.is_collision(pointUp))    or
            (dirRight and game.is_collision(pointDown)),
            # if there is danger to the left
            (dirDown  and game.is_collision(pointRight)) or
            (dirUp    and game.is_collision(pointLeft))  or
            (dirRight and game.is_collision(pointUp))    or
            (dirLeft  and game.is_collision(pointDown)),
            # add each direction's boolean value to show move direction
            dirLeft,
            dirRight, 
            dirUp, 
            dirDown, 
            # add the location of the food
            # for food at left
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        # return as array of ints
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # will pop from left if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # returns a list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        # now train step
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves
        # tradeoff between exploration and exploitation
        ep_games = 80
        self.epsilon = ep_games - self.n_games
        final_move = [0, 0, 0]
        # take random value if this case is true
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state_0)
            move = torch.argmax(prediction).item()
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        old_state = agent.get_state(game)
        # get move based on current state
        final_move = agent.get_action(old_state)
        # move and get the new state
        reward, done, score = game.play_step(final_move)
        # update new state
        new_state = agent.get_state(game)
        # train the short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        # remember all of this
        agent.remember(old_state, final_move, reward, new_state, done)
        # if we fully complete
        if done:
            # train the long memory and plot the result!
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            # check if there's a new high score
            if score > record:
                record = score
                # TODO: agent.model.save()
        print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')
        # TODO: Plot

if __name__ == '__main__':
    train()