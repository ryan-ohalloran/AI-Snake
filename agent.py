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
        # TODO: model, trainer


    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass

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