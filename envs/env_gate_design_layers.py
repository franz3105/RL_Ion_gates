import numpy as np

from envs.env_gate_design import IonGatesCircuit
from scipy.optimize import minimize
from joblib import Parallel, delayed
from operator import itemgetter
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba


# This class is another version of the compilation task. It is closer to layer-based compilation.
class IonGatesCircuitLayered(IonGatesCircuit):

    def __init__(self, **kwargs):
        self.ion_circuit = IonGatesCircuit(**kwargs)
        IonGatesCircuit.__init__(self, **kwargs)
        self.num_ms_gates = 7  # Number of MS gates
        self.n_loc_gates = 6  # Number of non-MS gates
        self.max_n_loc_gates = 5  # Maximum number of non-MS gates
        self.ms_pointer = 0  # Pointer defining the MS-layer
        self.loc_pointer = 0  # Pointer defining the non-MS-layer
        self.num_actions = self.num_actions - 1
        self.ms_loc_unitaries_dict = dict()
        self.create_action_hash()
        self.state_dimension = self.num_ms_gates * (self.n_loc_gates + 1)

        for i_ms in range(self.num_ms_gates + 1):
            self.ms_loc_unitaries_dict[f"MS{i_ms + 1}"] = np.zeros(self.n_loc_gates, np.int)

    def reset(self):

        """
        Standard reset function of the environment.
        :return: Environment state.
        """
        for i_ms in range(self.num_ms_gates + 1):
            self.ms_loc_unitaries_dict[f"MS{i_ms + 1}"] = np.zeros(self.n_loc_gates, np.int)
        self.ms_pointer = 0 # Resets pointer defining the MS-layer
        self.loc_pointer = 0 # Resets pointer defining the non-MS-layer

        return self.set_state()

    def set_state(self):

        """
        Sets the state of the environment.
        :return: A flattened array representing the state of the environment.
        """
        next_state_list = []
        for i_ms in range(self.num_ms_gates):
            loc_gates = self.ms_loc_unitaries_dict[f"MS{i_ms + 1}"]
            loc_gates = np.append(loc_gates, np.array([1], np.float64))
            next_state_list.append(loc_gates)

        return np.stack(next_state_list).flatten()

    def create_action_hash(self):
        self.action_dict = dict()

        for i_g, _ in enumerate(self.gate_names):
            self.action_dict[i_g] = np.array([i_g + 1, 0])

        # for i_a in range(1, self.num_actions): # If you want to add more actions, add them here.
        #    for i_b in range(1, i_a):
        #        self.num_actions += 1
        #        self.action_dict[self.num_actions - 2] = np.array([i_a + 1, i_b + 1])
        # self.num_actions = len(self.action_dict)

    def step(self, action, cost):

        """
        :param action: Index of the gate.
        :param cost: Cost function (infidelity) of the compilation task,
        with variational parameters and circuit structure as input.
        :return: State of the environment, reward, optimal angles for the current circuit, infidelity
        """

        num_gates, num_angles = self.step_no_opt(action)
        reward, done, opt_angles, err = self.minimize_cost_function(cost, num_gates, None, num_angles)

        return self.set_state(), reward, done, opt_angles, err

    def step_no_opt(self, action):

        """
        Step without optimization.
        :param action: Index of the gate
        :return: Number of gates on the circuit, number of angles of these gates.
        """

        self.ms_pointer = self.ms_pointer % (self.num_ms_gates + 1) + 1
        if self.ms_pointer == 1:
            self.loc_pointer = self.loc_pointer + 1

        gate_pointers = self.action_dict[action]
        # print(self.action_dict)
        # print(self.num_actions)

        action_position = np.count_nonzero(self.ms_loc_unitaries_dict[f"MS{self.ms_pointer}"])
        # print(action_position)
        for g in gate_pointers:
            if action_position < self.n_loc_gates:
                action_position = np.count_nonzero(self.ms_loc_unitaries_dict[f"MS{self.ms_pointer}"])
                self.ms_loc_unitaries_dict[f"MS{self.ms_pointer}"][action_position] = int(g)
            else:
                continue

        next_state_list = []

        for i_ms in range(1, self.ms_pointer + 1):
            loc_gates = self.ms_loc_unitaries_dict[f"MS{i_ms}"]
            if i_ms != self.num_ms_gates:
                loc_gates = np.append(loc_gates, np.array([1]))
            else:
                loc_gates = np.append(loc_gates, np.array([0]))

            # print(loc_gates)
            next_state_list.append(loc_gates)

        # print(next_state_list)
        self.next_state = np.stack(next_state_list).flatten()
        self.next_state = self.next_state[self.next_state != 0]
        # print(self.next_state)
        # print(self.next_state)
        self.gate_sequence = list()
        # print(self.next_state)
        num_gates = np.count_nonzero(self.next_state)
        # print(num_gates)
        num_angles = 0
        for i_gate in range(num_gates):
            gate_type_number = int(self.next_state[i_gate]) - 1
            self.gate_sequence.append(self.gate_names[gate_type_number])
            if self.gate_names[gate_type_number] in ["MS", "Cxy"]:
                num_angles += 2
            elif self.gate_names[gate_type_number] in ["R_xyz_1", "R_xyz_2", "R_xyz_3"]:
                num_angles += 3
            else:
                num_angles += 1

        return num_gates, num_angles

    def minimize_cost_function(self, cost, num_gates, reduced, num_angles):

        """
        Minimize the cost function.
        :param cost: Cost function (infidelity) of the compilation task,
        :param num_gates: Number of gates on the circuit.
        :param reduced: Reduced circuit structure.
        :param num_angles: Number of angles of the gates.
        :return: Reward, whether the episode is done or not, optimal angles for the current circuit, infidelity
        """

        if self.loc_pointer > 0:
            # angle_init = np.random.randn(num_angles)
            # self.verify_gradient(angle_init, num_gates)

            def call(x):
                print(x)

            def cg(x):
                return cost(self.next_state[:num_gates], x, num_gates,
                            np.asarray(self.current_operation, np.complex128))

            start_value = 2 * np.pi * np.random.uniform(0, 1, size=(self.n_shots, num_angles))

            def min_fid(i):
                res = minimize(cg, start_value[i, :], method='l-bfgs-b',
                               options={'disp': False, 'maxiter': self.opt_iter},
                               jac=True)
                return res.fun, res.x

            result = Parallel(n_jobs=int(self.num_cores / 2))(delayed(min_fid)(i) for i in range(self.n_shots))
            err_list = [r[0] for r in result]
            print(min(err_list))
            angle_list = [a[1] for a in result]
            err_index, err = min(enumerate(err_list), key=itemgetter(1))
            opt_angles = angle_list[err_index]

        else:
            err = 1.
            opt_angles = np.zeros(num_angles)

        ms_count = 0
        for g in self.gate_sequence:
            if g in ["MS", "MSx", "MSy"]:
                ms_count += 1

        reward, done = self.curriculum.reward(err, ms_count, self.n_step) # Curriculum defines the reward function,
        # based on the error.
        self.threshold = self.curriculum.threshold # Updates the threshold

        if err < self.min_threshold:
            done = True
            print(self.gate_sequence)
            print(opt_angles)
            print("Found!")

        if self.loc_pointer >= self.n_loc_gates - 1 or self.action_counter > self.max_num_actions:
            done = True

        return reward, done, opt_angles, err
