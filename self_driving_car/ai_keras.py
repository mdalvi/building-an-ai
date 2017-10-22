# coding: utf-8

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# https://docs.python.org/2/library/collections.html#collections.deque
from collections import deque


class Model(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        # # Neural Network for Deep Q Learning
        #
        # # Sequential() creates the foundation of the layers
        # model = Sequential()
        #
        # # 'Dense' is the basic form of a neural network layer
        # # Input Layer of state size(5) and Hidden Layer with 30 nodes
        # model.add(Dense(units=30, input_dim=self.state_size, activation='relu'))
        #
        # # Hidden layer with 30 nodes
        # model.add(Dense(units=30, activation='relu'))
        #
        # # Output Layer with # of actions: 3 nodes (straight, left, right)
        # model.add(Dense(units=self.action_size, activation='linear'))
        #
        # # Create the model based on the information above
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):  # a.k.a. remember function
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Dqn(object):
    def __init__(self, state_size, action_size, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.model = Model(state_size=state_size, action_size=action_size)
        self.memory = Memory(capacity=100000)
        self.last_state = np.zeros((1, 5))
        self.last_action = 0
        self.last_reward = 0.0
        self.reward_bag = deque(maxlen=1000)
        self.study_time = 500
        self.episode = 0

        # if using the epsilon-greedy action selection policy
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):

        # if using the epsilon-greedy action selection policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

        # if using the softmax action selection policy
        # return self.sampler(self.softmax(X=np.array([act_values[0]]), theta=10.0, axis=1)[0])

    def sampler(self, distrbution):
        r = random.random()
        cumulative = 0.0
        action = 0
        for probability in distrbution:
            cumulative += probability
            if r < cumulative:
                return action
            action += 1

    def learn(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state in minibatch:
            # the original Q-learning formula
            # http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

            # first feed forward pass to get the predicted Q-values of all actions
            target_predicted = self.model.predict(state)

            # second feed forward pass to get the maximum Q-value over all network outputs
            target_q = (reward + (self.gamma * np.amax(self.model.predict(next_state)[0])))

            # setting the Q-value for the action a to target_q. 
            # For all other action, set the Q-value target to the same as originally returned from step 1, making the error 0 for those outputs
            target_predicted[0][action] = target_q

            self.model.model.fit(state, target_predicted, epochs=1, verbose=0)

            # if using the epsilon-greedy action selection policy
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def update(self, reward, new_state):
        new_state = np.array([new_state])
        self.memory.push(self.last_state, self.last_action, self.last_reward, new_state)
        action = int(self.act(new_state))
        if len(self.memory) > 1000 and self.study_time <= 0:
            self.learn(batch_size=1000)
            self.study_time = 500
            self.episode += 1
            print("EPISODE: {}, SCORE: {}".format(self.episode, (sum(self.reward_bag) / (len(self.reward_bag) + 1.0))))

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_bag.append(reward)
        self.study_time -= 1
        return action

    def score(self):
        return (sum(self.reward_bag) / (len(self.reward_bag) + 1.0))

    def load(self, name):
        self.model.load(name)

    def save(self, name):
        self.model.save(name)

    # if using the softmax action selection policy
    # as found on https://nolanbconaway.github.io/blog/2017/softmax-numpy
    def softmax(self, X, theta=1.0, axis=None):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

        # make X at least 2d
        y = np.atleast_2d(X)

        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        # multiply y against the theta parameter,
        y = y * float(theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)

        # exponentiate y
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

        # finally: divide elementwise
        p = y / ax_sum

        # flatten if X was 1D
        if len(X.shape) == 1: p = p.flatten()

        return p
