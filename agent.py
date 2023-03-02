import random
import time as tm
from collections import deque

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical

class DQNAgent:
    def __init__(self, action_size, learning_rate, model: Model, get_legal_actions):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = learning_rate
        self.model = model
        self.get_legal_actions = get_legal_actions

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        epsilon = self.epsilon
        bestAction = self.get_best_action(state)

        if random.uniform(0, 1) < epsilon:
            temp=[a for a in possible_actions if a != bestAction]
            if len(temp)>0:
                return random.choice(temp)
        return bestAction

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None
        state = np.array([to_categorical(state, num_classes=1296)])
        return np.argmax(self.model.predict(state))

    def lower_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon >= self.epsilon_min:
            self.epsilon = new_epsilon

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        info_sets = random.sample(self.memory, batch_size)
        states_list = []
        targets_list = []
        for info_set in info_sets:
            state, action, reward, next_state, done = info_set
            states_list.append(state.flatten())
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            targets_list.append(target.flatten())

        states_array = np.array(states_list)
        targets_array = np.array(targets_list)

        self.model.train_on_batch(states_array, targets_array)
        self.lower_epsilon()
