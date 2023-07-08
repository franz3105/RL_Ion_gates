import numpy as np
from collections import deque

# This class creates a curriculum learning schedule for the reinforcement learning algorithm or the beam search
# algorithm. It progressively lowers the threshold for the cost function, so that the algorithm is forced to learn
# the easier tasks first. Based on the agent's performance on the previous task, the threshold is lowered or raised.
# The threshold is lowered if the agent performs well on the previous task, and raised if the agent performs poorly.

class Curriculum:

    def __init__(self, start_threshold, max_t_steps, curr_window, min_threshold=0.01, delta=0.5):

        """

        :param start_threshold: starting threshold for the curriculum learning
        :param max_t_steps: maximum number of time steps per episode
        """
        self.err_memory = deque(maxlen=curr_window)
        self.err_memory.append(1)
        self.min_threshold = min_threshold
        self.threshold = start_threshold
        self.err_check = curr_window
        self.s_line = np.arange(0, 1, 10)
        self.best_counter = 0
        self.delta = delta
        self.max_t_steps = max_t_steps

    def reward(self, err, ms_count, t):
        """
        :param err: error of the current episode
        :param ms_count:
        :param t:
        :return:
        """

        done = False

        if err < self.threshold:
            self.best_counter += 1
            reward = 2
            done = True

            if err < self.min_threshold:
                reward = 10
                done = True
        else:
            if t % (self.max_t_steps - 2) == 0:
                reward = 0
            else:
                reward = 0

        self.store(err, t)
        return reward, done

    def update_threshold(self):
        """
        Update the threshold for the curriculum learning.
        :return:
        """
        min_err = np.clip(min(self.err_memory), self.min_threshold, 1)

        if min_err < self.threshold:
            self.threshold = np.clip(min_err + self.delta * (self.threshold - min_err), self.min_threshold, 1)

        if min_err == self.threshold:
            self.threshold = self.threshold * 0.99

        return self.threshold

    def store(self, err, t):
        """
        Store the error of the current episode in the memory.
        :param err:
        :param t:
        :return:
        """
        self.err_memory.append(err)

        if self.best_counter > self.err_check:
            self.best_counter = 0
            self.update_threshold()

        return
