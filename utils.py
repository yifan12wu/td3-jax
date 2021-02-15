import time

import numpy as np
import jax
from jax import random as jrandom


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            jax.device_put(self.state[ind]),
            jax.device_put(self.action[ind]),
            jax.device_put(self.next_state[ind]),
            jax.device_put(self.reward[ind]),
            jax.device_put(self.not_done[ind])
        )


class Timer:

    def __init__(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def reset(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def set_step(self, step):
        self._step = step
        self._step_time = time.time()

    def time_cost(self):
        return time.time() - self._start_time

    def steps_per_sec(self, step):
        sps = (step - self._step) / (time.time() - self._step_time)
        self._step = step
        self._step_time = time.time()
        return sps


class PRNGKeys:

    def __init__(seed=0):
        self._key = jrandom.PRNGKey(seed)

    def get_key():
        self._key, subkey = jrandom.split(self._key)
        return subkey