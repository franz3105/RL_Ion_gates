import unittest
import numpy as np

from envs.env_gate_design import IonGatesCircuit
from envs.env_gate_design_layers import IonGatesCircuitLayered
from data.env_state_design import IonStatesCircuit


class EnvTest(unittest.TestCase):

    def test_env(self):

        env = IonGatesCircuit(4, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)
        env.step(3)
        env.step(4)
        self.assertTrue(env.next_state is not None)
        self.assertTrue(env.next_state[:4] == np.array([1, 2, 3, 4]))

        env = IonGatesCircuitLayered(4, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)
        env.step(3)
        self.assertTrue(env.next_state is not None)
        self.assertTrue(env.next_state[:4] == np.array([1, 2, 3, 4]))

        env = IonStatesCircuit(4, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)
        env.step(3)
        self.assertTrue(env.next_state is not None)
        self.assertTrue(env.next_state[:4] == np.array([1, 2, 3, 4]))



if __name__ == '__main__':
    unittest.main()
