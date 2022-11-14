from abc import ABC

import gym
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from PSO.pso_function import PSOEnvironment
import numpy as np

class GymPSO(gym.Env, ABC):

    def __init__(self, resolution, ys, method, reward_function='mse', initial_seed=1000, behavioral_method=0):


        self.env = PSOEnvironment(resolution=resolution,
                                  ys=ys,
                                  method=method,
                                  reward_function=reward_function,
                                  initial_seed=initial_seed,
                                  behavioral_method=behavioral_method,)

        """ Action space - Always the same """
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,))

        """ Observation space - depend on method"""
        self.observation_space = gym.spaces.Box(-10, 10, shape=self.env.state.shape)

        self.method = method

    @staticmethod
    def process_state(state, method):

        """ Vector like"""
        if method == 0:
            return state/100.0
        elif method == 1:

            for i in range(4):
                state[i] += state[i]

        else:
            raise NotImplementedError

        return state

    def reset(self):

        state = self.env.reset()

        return self.process_state(state, self.method)

    def seed(self, seed=None):

        return np.random.seed(seed)

    def step(self, action):

        action_denormalized = (action + 1)/2 * 4

        state, reward, done, _ = self.env.step(action_denormalized)

        return self.process_state(state, self.method), reward, done, {}






