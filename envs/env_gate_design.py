import jax
import numpy as np
import psutil
import multiprocessing as mp

from optimizers.jax_minimize_wrapper import minimize_jax
from scipy.optimize import minimize
from numpy_ringbuffer import RingBuffer
from operator import itemgetter
from envs.curriculum_learning import Curriculum
from envs.env_utils import tensor, construct_cost_function
from joblib import Parallel, delayed


# This is the standard class for the gate design problem.

class IonGatesCircuit(object):

    def __init__(self, target_gate, num_qubits, gate_names, x_opt, max_len_sequence=40, state_output="circuit",
                 simplify_state=True, seed=0, threshold=0.1, min_gates=1, n_shots=1, library="numba", max_iter=100,
                 min_threshold=1e-2, curriculum_window=500):

        """
        Environment for compilation of trapped-ion gates.
        :param target_gate: Target unitary for the compilation
        :param num_qubits: Number of qubits.
        :param gate_names: Labels describing the gates.
        :param x_opt: Optimizer to use.
        :param max_len_sequence: Maximal length of the sequence.
        :param state_output: The type of output of the environment state.
        :param simplify_state: Whether nearest-neighbour gates (e.g. XX = I) should be simplified or not.
        :param seed: Seed for the random number generator.
        :param threshold: Reward threshold.
        :param min_gates: Minimum number of gates to consider for the reward.
        :param n_shots: Number of optimization trajectories.
        :param library: Library to use for the calculation of cost and gradient.
        :param max_iter: Maximum number of iterations.
        :param min_threshold: Minimum threshold for the reward.
        :param curriculum_window: Number of steps in the curriculum learning window.
        """

        np.random.seed(seed)

        self.num_qubits = num_qubits  # Number of qubits
        self.d = 2  # Dimension of the Hilbert space (qudit)

        self.state_output = state_output  # The type of output of the state
        self.x_opt = x_opt
        self.opt_iter = max_iter

        self.n_shots = n_shots  # Number of optimization trajectories
        self.threshold = threshold
        self.min_gates = min_gates
        self.max_num_actions = 2 * max_len_sequence  # Maximum number of actions
        self.max_len_sequence = max_len_sequence  # Maximum length of the gate sequence
        self.next_state = np.zeros(self.max_len_sequence, np.int) # Initial state
        self.time_window = 1
        self.ms_count = 0
        self.current_operation = np.identity(self.d ** self.num_qubits, np.complex128)  # identity on all qubits
        self.gate_sequence = [] # Gate sequence
        self.max_ms = 5  # Maximum number of MS gates
        self.n_step = 0 # Number of steps in the curriculum learning window
        self.min_threshold = min_threshold # Minimum threshold for the reward
        self.curr_window = curriculum_window # Number of steps in the curriculum learning window
        # Gate set and cost_grad function setup
        self.library = library # Library to use for the calculation of cost and gradient
        self.minimize_function = minimize if self.library == "numba" else minimize_jax

        if len(target_gate.shape) < 2:
            self.time_dep_u = False
        else:
            self.time_dep_u = True

        self.target_gate = target_gate
        self.gate_names = gate_names

        self.use_curriculum = True
        self.state_output = state_output
        self.action_counter = 0
        self.good_guess_dict = dict()

        self.action_dict = dict()
        self.num_actions = len(self.gate_names)
        self.action_upper_bound = max(self.num_actions, 10)
        self.curriculum = Curriculum(self.threshold, self.max_len_sequence, self.curr_window, self.min_threshold)

        # LSTM state setup
        self.sequence_count = 1
        self.state_list = RingBuffer(capacity=self.sequence_count, dtype=(np.float32, self.time_window))
        for i in range(self.sequence_count):
            self.state_list.append(self.next_state.copy()[-self.time_window:])

        self.num_percepts_list = np.full([self.max_len_sequence], len(self.gate_names))
        self.create_action_hash()
        self.two_angle_gates = ["MS", "Cxy"]
        self.init_unitary = np.identity(self.d ** self.num_qubits, np.complex128)
        self.prev_err = 1
        self.prev_angles = np.zeros(2)

        self.episode_rewards = []

        self.best_fidelity = 0.
        self.start_values = np.random.randn(self.n_shots, self.max_len_sequence * 3)

        self.angle_population = np.random.randn(2 * max_len_sequence)
        # print(self.num_actions)
        if self.state_output == "circuit":
            self.state_dimension = self.max_len_sequence * self.action_upper_bound
        elif self.state_output == "lstm_circuit":
            self.state_dimension = self.time_window * self.action_upper_bound
        elif self.state_output == "unitary":
            self.state_dimension = 2 * (self.d ** (2 * self.num_qubits))
        elif self.state_output == "state":
            self.state_dimension = 2 * self.d ** self.num_qubits
        # self.cg_numba = create_cost_gates_standard(self.num_gates, self.target, *self.gate_functions)

        self.num_cores = psutil.cpu_count()
        self.simplify_state = simplify_state

    def set_state(self):
        """
        There are different possible output modes for the state. The structure of the circuit, the unitary (or state),
        the action at time t for  the LSTM circuit.
        :return:
        """
        if self.state_output == "unitary":
            rl_state = np.array([np.real(self.current_operation), np.imag(self.current_operation)],
                                dtype=np.float64).flatten()
        elif self.state_output == "rl_state":
            rl_state = np.array([np.real(self.current_state), np.imag(self.current_state)],
                                dtype=np.float64).flatten()
        elif self.state_output == "circuit":
            # print(self.encode_state(self.next_state))
            rl_state = self.encode_state(self.next_state)
        elif self.state_output == "lstm_circuit":
            rl_state = self.encode_lstm_state(self.state_list)
        else:
            raise NotImplementedError(f"The rl_state output mode \"{self.state_output}\" is not implemented")

        # print(rl_state)
        return rl_state

    def get_action_position(self):
        """
        Get the position of the action in the state
        :return: The last non-empty position of the quantum circuit.
        """

        action_position = 0
        for i, el in enumerate(self.next_state):
            if np.equal(el, [0, 0]).all():
                action_position = i
                break
        return action_position

    def state_simplification(self):

        """
        This functions simplifies the state by removing two repeated gates if they appear next to each other.
        E.g. XX = I etc. It simplifies the training of the agent.
        :return:
        """
        state = np.array(self.next_state, dtype=np.int)
        # print(state)
        action_position = self.get_action_position()
        action_before = state[action_position - 2]
        action_after = state[action_position - 1]

        gate_type_before = self.gate_names[int(action_before) - 1]
        gate_type_after = self.gate_names[int(action_after) - 1]

        # if gate_type_after == "MS":
        #    ms_count = self.get_ms_gates()
        #    if ms_count > self.max_ms - 1:
        #        state[action_position - 1] = 0
        reduced = False

        if gate_type_before == gate_type_after:  # and gate_type_after not in ["MS", "Cxy"]:
            state[action_position - 1] = 0
            state[action_position - 2] = action_before
            reduced = True

        return state, reduced

    def reset(self):

        """
        Standard reset function for the environment.
        :return:
        """
        self.current_operation = tensor([np.eye(2)] * self.num_qubits)
        self.current_state = np.zeros([2 ** self.num_qubits, 1])
        self.current_state[0, 0] = 1
        self.gate_sequence = []
        self.action_counter = 0
        self.next_state = np.zeros(self.max_len_sequence, np.int)
        self.angle_population = np.random.randn(2 * self.max_len_sequence)
        for i in range(self.sequence_count):
            self.state_list.append(self.next_state.copy()[-self.time_window:])
        self.start_values = np.random.randn(self.n_shots, self.max_len_sequence * 3)
        self.prev_err = 1
        self.prev_angles = np.zeros(2)
        return self.set_state()
        # print(self.gate_sequence)

    def encode_lstm_state(self, state):

        """
        Encoded representation of the state for the LSTM agent.
        :param state:
        :return:
        """
        encoded_rep = np.zeros([self.time_window, self.action_upper_bound])
        # print(np.array(state))
        for i_t in range(self.time_window):
            encoded_rep[i_t, np.int(state[i_t]) - 1] = 1

        return encoded_rep

    def encode_state(self, state):

        """
        Encodes the state in a two-hot encoding.
        :param state: Array of numbers describing the structure of the circuit, where the index is the position of the
        gate in the circuit and the number is the index of the gate in the gate set.
        :return:
        """
        encoded_rep = np.zeros([self.max_len_sequence, self.action_upper_bound])
        action_pointer = np.count_nonzero(state)
        # print(state)
        autoreg_input = state.copy()
        if action_pointer > self.time_window:
            autoreg_input[0:action_pointer - self.time_window] = 0

        for i_el, el in np.ndenumerate(autoreg_input[:action_pointer]):
            if el != 0:
                encoded_rep[i_el, el - 1] = 1
        # print(encoded_rep)
        return encoded_rep

    def search_for_Z_gates(self):

        """
        Searches for Z gates in the current circuit.
        :return:
        """
        indices = []

        for i_pointer, gate_pointer in self.next_state:
            if gate_pointer > 1:
                indices.append([gate_pointer, i_pointer])

        return indices

    def numericalgradient(self, cost_grad, angles, action_position):

        """
        Numerical gradient of the cost function for comparison.
        :param cost_grad: Cost function and gradient.
        :param angles: Cost function input, gate rotation angles.
        :param action_position: The length of the actual circuit.
        :return:
        """

        def new_cg(x):
            return cost_grad(self.next_state[:action_position], x, action_position)

        n_DOF = len(angles)
        # print(n_DOF)
        dx = np.zeros(n_DOF)
        err = new_cg(angles, action_position)[0]
        # print("err ", err)
        for k in range(0, n_DOF):
            angles2 = np.zeros(n_DOF)
            for k2 in range(0, n_DOF):
                angles2[k2] = angles[k2]
            angles2[k] = angles[k] + 1e-9
            dx[k] = -(err - new_cg(angles2, action_position)[0]) * 1e9
        return dx

    def verify_gradient(self, cost_grad, angles, action_position):
        """
        Compares the numerical gradient of the cost function with the analytic gradient.
        :param cost_grad: Cost function and gradient.
        :param angles: Cost function input, gate rotation angles.
        :param action_position: The length of the actual circuit.
        :return:
        """

        def new_cg(x):
            return cost_grad(self.next_state[:action_position], x, action_position)

        print("\n\nthe error: ", new_cg(angles, action_position)[0])
        GR = new_cg(angles, action_position)[1]
        GR2 = self.numericalgradient(angles, action_position)
        print("its analytic gradient: ", GR)
        print("its numerical gradient: ", GR2)
        print("its gradient error: ", GR - GR2)

    def create_action_hash(self):
        self.action_dict = dict()

        for i_g, _ in enumerate(self.gate_names):
            self.action_dict[i_g] = np.array([i_g + 1, 0])

        # for i_a in range(self.num_actions):
        #    for i_b in range(i_a):
        #        self.num_actions += 1
        #        self.action_dict[self.num_actions - 1] = np.array([i_a + 1, i_b + 1])
        # print(self.action_dict)

    def angle_count(self):

        """
        Counts the number of parameters needed to the variational circuit.
        :return: Number of gates on the circuit, number of parameters, sequence of gates as strings.
        """
        num_gates = np.count_nonzero(self.next_state)
        num_angles = 0

        for i_gate in range(num_gates):
            gate_type_number = int(self.next_state[i_gate]) - 1
            self.gate_sequence.append(self.gate_names[gate_type_number])
            if self.gate_names[gate_type_number] in self.two_angle_gates:
                num_angles += 2
            elif self.gate_names[gate_type_number] in ["R_xyz_1", "R_xyz_2", "R_xyz_3"]:
                num_angles += 3
            else:
                num_angles += 1

        return num_gates, num_angles, self.gate_sequence

    def get_ms_gates(self):

        """
        Counts the number of MS gates on the circuit.
        :return: Number of MS gates on the circuit.
        """
        ms_count = 0
        for g in self.gate_sequence:
            if g in ["MS", "MSx", "MSy"]:
                ms_count += 1
        return ms_count

    def minimize_numba(self, cost, num_gates, num_angles):
        """

        :param cost:
        :param num_gates:
        :param num_angles:
        :return:
        """

        def cg(x):
            return cost(self.next_state[:num_gates], x, num_gates,
                        np.asarray(self.current_operation, np.complex128))

        num_params = num_angles
        self.start_values = 2 * np.pi * np.random.randn(self.n_shots, num_params)

        def min_fid(i):
            res = self.minimize_function(cg, self.start_values[i, :num_params], method='l-bfgs-b',
                                         options={'disp': False, 'maxiter': 100},
                                         jac=True)
            return res.fun, res.x

        result = Parallel(n_jobs=int(self.num_cores / 2))(delayed(min_fid)(i)
                                                          for i in range(self.n_shots))
        err_list = [r[0] for r in result]
        print(min(err_list))
        # print(err_list)
        angle_list = [a[1] for a in result]
        angle_arr = np.array(angle_list)

        # print(err_list[0], check_list[best_idx])
        # self.start_values[:, :angle_arr.shape[1]] = angle_arr
        err_index, err = min(enumerate(err_list), key=itemgetter(1))
        # print(err)
        opt_angles = np.array(angle_arr[err_index])

        return err, opt_angles

    def minimize_jax(self, cost, num_gates):

        """

        :param cost:
        :param num_gates:
        :return:
        """
        num_params = num_gates * 2

        self.start_values = 2 * np.pi * np.random.randn(self.n_shots, num_params)

        def cg(x):

            return cost(self.next_state[:num_gates], x, num_gates)

        def min_fid(i):
            res = self.minimize_function(cg, self.start_values[i, :num_params], method='l-bfgs-b',
                                         options={'disp': False, 'maxiter': self.opt_iter},
                                         jac=True)
            return [res.fun], [res.x]

        if self.n_shots == 1:
            result = min_fid(0)
        else:
            result = self.x_opt(self.next_state[:num_gates], self.start_values[:, :num_params], num_gates)

        err_list = result[0]

        angle_list = result[1]
        angle_arr = np.array(angle_list)

        err_index, err = min(enumerate(err_list), key=itemgetter(1))
        opt_angles = np.array(angle_arr[err_index])
        return err, opt_angles

    def minimize_cost_function(self, cost, num_gates, reduced, num_angles):

        """
        Minimize the cost function.
        :param cost: Cost function (infidelity) of the compilation task,
        :param num_gates: Number of gates on the circuit.
        :param reduced: Reduced circuit structure.
        :param num_angles: Number of angles of the gates.
        :return: Reward, whether the episode is done or not, optimal angles for the current circuit, infidelity
        """

        reward = 0
        done = False

        if (num_gates + 1) % 1 == 0 and not reduced:

            if self.library == "numba":

                err, opt_angles = self.minimize_numba(cost, num_gates,
                                                      num_angles)  # Uses scipy.optimize.minimize, numba,
                # and parallelization
            elif self.library == "jax":
                err, opt_angles = self.minimize_jax(cost, num_gates)  # Uses jaxopt, jax, and parallelization
            else:
                raise ValueError("Library not supported")

            self.prev_err = err
            self.prev_angles = opt_angles

        else:
            err = self.prev_err
            opt_angles = self.prev_angles

        ms_count = 0
        for g in self.gate_sequence:
            if g in ["MS", "MSx", "MSy"]:
                ms_count += 1

        if self.use_curriculum:
            reward, done = self.curriculum.reward(err, ms_count, num_gates)
            #reward = max(reward - 0.5 * ms_count - 0.01 * len(self.gate_sequence), 0)

            self.threshold = self.curriculum.threshold
            # print(self.gate_sequence)

            if self.prev_err is None:
                self.prev_err = err

            if err < self.min_threshold:
                # reward = - np.log10(err) - 0.01*ms_count - 0.01*len(self.gate_sequence)
                print(self.gate_sequence)
                print("Found!")
                # done = True

            if num_gates >= self.max_len_sequence - 2 or self.action_counter > self.max_num_actions:
                done = True

        else:
            if err < self.min_threshold:
                reward = - np.log10(err) - 0.01 * ms_count - 0.01 * len(self.gate_sequence)
                print(self.gate_sequence)
                print("Found!")
                done = True

            if num_gates >= self.max_len_sequence - 2 or self.action_counter > self.max_num_actions:
                reward = 0  #
                done = True

        return reward, done, opt_angles, err

    def step_no_opt(self, action):

        """
        Step without optimization (e.g. for discrete gates)
        :param action: Next gate on the circuit.
        :return:
        """
        gate_pointers = self.action_dict[action]

        # compute the new operation
        action_position = np.count_nonzero(self.next_state)

        self.current_operation = tensor([np.eye(2, dtype=np.complex128)] * self.num_qubits)
        self.n_step = self.n_step + 1

        for g in gate_pointers:
            self.next_state[action_position] = int(g)
            action_position = np.count_nonzero(self.next_state)

        # print(self.next_state)
        reduced = False
        self.action_counter += 1

        if self.simplify_state:
            if action_position > 0:
                self.next_state, reduced = self.state_simplification()

        if action_position > self.time_window - 1:
            self.state_list.append(self.next_state.copy()[action_position - self.time_window:action_position])
        else:
            self.state_list.append(self.next_state.copy()[:self.time_window])

        self.gate_sequence = list()
        num_gates = np.count_nonzero(self.next_state)
        num_angles = 0

        for i_gate in range(num_gates):
            gate_type_number = int(self.next_state[i_gate]) - 1
            self.gate_sequence.append(self.gate_names[gate_type_number])
            if self.gate_names[gate_type_number] in self.two_angle_gates:
                num_angles += 2
            elif self.gate_names[gate_type_number] in ["R_xyz_1", "R_xyz_2", "R_xyz_3"]:
                num_angles += 3
            else:
                num_angles += 1

        return num_gates, reduced, num_angles

    def step(self, action, cost):

        """

        :param action:
        :param cost:
        :return:
        """

        num_gates, reduced, num_angles = self.step_no_opt(action)
        reward, done, opt_angles, err = self.minimize_cost_function(cost, num_gates, reduced, num_angles)

        return self.set_state(), reward, done, opt_angles, err


class MultiIonGatesCircuit:

    def __init__(self, envs, cost_grad, **kwargs):
        self.cost_grad = cost_grad
        self.envs = envs

    def ev_cost_grad(self, vec_next_state, vec_angles, action_position):
        return jax.vmap(lambda a, b: self.cost_grad(a, b, action_position))(vec_next_state, vec_angles)

    def multi_step(self, actions):
        n_processes = 6
        with mp.Pool(n_processes) as p:
            p.map(lambda e, a: e.step_no_opt(a), (self.envs, actions))
