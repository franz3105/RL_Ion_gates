import numpy as np
import psutil
import os
import datetime
import time

from envs.env_gate_design import IonGatesCircuit
from joblib import Parallel, delayed
from operator import itemgetter
from quantum_circuits.cost_and_grad_numba import create_cost_gates_standard
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from envs.env_utils import construct_cost_function


# This class is used to compile a circuit in layers, but does not search through the possible gate combinations.
# Nonetheless, it can always find you a circuit that is better than the one you started with, even if it is not the
# best possible circuit.

class LayerCompilation(IonGatesCircuit):

    def __init__(self, num_qubits, target_gate, library, structure=False, max_iter=100, **kwargs):

        """
        Constructs a class to perform layer-based gradient-based compilation

        :param num_qubits: Number of qubits on the circuit.
        :param target_gate: Target unitary.
        :param library: Type of library used ("jax" or "numba")
        :param kwargs: Other parameters of IonGatesCircuit.
        """

        super().__init__(num_qubits=num_qubits, target_gate=target_gate, **kwargs)
        # self.env = IonStatesCircuit(**kwargs)
        self.max_n_iter = max_iter
        self.max_n_gates = self.max_len_sequence
        self.next_state = np.zeros(self.max_n_gates + self.num_qubits + 2, np.int32)
        self.num_ms = 0
        gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries = construct_cost_function("standard",
                                                                                                           library,
                                                                                                           num_qubits,
                                                                                                           target_gate,
                                                                                                           time_dep_u=True)
        self.x_opt = x_opt
        self.library = library
        self.gate_funcs = gate_funcs
        self.cost = vec_cost_grad
        self.cs_to_unitaries = cs_to_unitaries
        self.U0 = np.identity(2 ** num_qubits, np.complex128)
        self.num_cores = psutil.cpu_count()
        self.target_gate = self.target_gate
        self.gate_functions, self.gate_names, self.is_layered = create_fast_ms_set_numba(self.num_qubits)
        self.cg_numba = create_cost_gates_standard(self.num_qubits, self.target_gate, *self.gate_functions)[1]
        self.errors = []
        self.number_of_gates = []
        self.angles = []
        self.gate_sequences = []
        print(self.n_shots)

    def initialize_circuit(self):

        """
        Initializes the circuit as a general rotation (no entanglement).
        :return: Index defining the size of the circuit.
        """

        self.next_state = np.zeros(self.max_n_gates, np.int32)
        state_pointer = 0
        state_pointer = self.apply_local_gate(state_pointer)
        self.next_state[state_pointer] = 1
        state_pointer += 1
        state_pointer = self.apply_local_gate(state_pointer)

        return state_pointer

    def apply_local_gate(self, state_pointer):

        """
        Applies a rotation R_xy Z_1 ... Z_n R_xy
        :param state_pointer: Index defining the size of the circuit.
        :return: Updated state pointer.
        """

        self.next_state[state_pointer] = 2
        state_pointer += 1
        for i_z in range(self.num_qubits):
            self.next_state[state_pointer] = 3 + i_z
            state_pointer += 1
        self.next_state[state_pointer] = 2
        state_pointer += 1

        return state_pointer

    def save_results(self, folder):

        """
        Saves the results of the compilation.
        :param folder: System path.
        """
        cwd = os.getcwd()
        now = datetime.datetime.now()

        if folder:
            data = os.path.join(cwd, folder)
            if not os.path.exists(data):
                os.mkdir(data)
            agent_dir = os.path.join(data, "data_struct" + now.strftime("%m_%d_%Y_%H_%M_%S"))
            if not os.path.exists(agent_dir):
                os.mkdir(agent_dir)
        else:
            data = os.path.join(cwd, "data_rl")
            now = datetime.datetime.now()
            if not os.path.exists(data):
                os.mkdir(data)
            agent_dir = os.path.join(data, "data_struct" + now.strftime("%m_%d_%Y_%H_%M_%S"))
            if not os.path.exists(agent_dir):
                os.mkdir(agent_dir)

        np.savetxt(os.path.join(agent_dir, "fidelities.txt"), 1 - np.asarray(self.errors))
        np.savetxt(os.path.join(agent_dir, "circuit_length.txt"), np.asarray(self.number_of_gates))
        np.savetxt(os.path.join(agent_dir, "angle_params.txt"), np.concatenate(self.angles).flatten())

        with open(os.path.join(agent_dir, "sequences.txt"), "w+") as f:
            for i_seq, seq in enumerate(self.gate_sequences):
                f.write("".join(seq))

        # with open(os.path.join(agent_dir, 'commandline_args.txt'), 'w') as f:
        #    json.dump(self.__dict__, f, indent=2)
        #    f.write(f"state = {self.target}\n")

    def minimize_cost_(self, cost, num_gates, circuit, angles_0):

        """
        Minimizes the cost function using gradient descent.
        :param cost: Cost function.
        :param num_gates: Number of quantum_circuits on the circuit.
        :param num_angles: Number of angles on the circuit.
        :return: Tuple with the best angles and the best cost.
        """
        t0 = time.time()
        num_params = angles_0.shape[1]

        if self.library == "numba":

            def cg(x):

                return cost(circuit, x, num_gates, self.current_operation)
            print(cg(angles_0[0, :num_params]))

            def min_fid(i):
                res = self.minimize_function(cg, angles_0[i, :num_params], method='l-bfgs-b',
                                             options={'disp': False, 'maxiter': 100},
                                             jac=True)
                return res.fun, res.x

            result = Parallel(n_jobs=int(self.num_cores / 2))(delayed(min_fid)(i)
                                                              for i in range(self.n_shots))
            err_list = [r[0] for r in result]
            # print(min(err_list))
            angle_list = [a[1] for a in result]
            angle_arr = np.array(angle_list)

            err_index, err = min(enumerate(err_list), key=itemgetter(1))
            opt_angles = np.array(angle_arr[err_index])
        else:

            def cg(x):

                return cost(circuit, x, num_gates)

            print(cg(angles_0[0, :num_params])[1].shape)
            print(angles_0[0, :num_params].shape)

            def min_fid(i):
                res = self.minimize_function(cg, angles_0[i, :num_params], method='l-bfgs-b',
                                             options={'disp': False, 'maxiter': self.max_n_iter},
                                             jac=True)
                return [res.fun], [res.x]

            if self.n_shots == 1:
                result = min_fid(0)
            else:
                result = self.x_opt(circuit, angles_0[:, :num_params], num_gates)

            err_list = result[0]
            # print(min(err_list))

            angle_list = result[1]
            angle_arr = np.array(angle_list)

            err_index, err = min(enumerate(err_list), key=itemgetter(1))
            opt_angles = np.array(angle_arr[err_index])

        t1 = time.time() - t0
        self.prev_err = err
        self.prev_angles = opt_angles

        return err, opt_angles, t1

    def run_compilation(self):

        """
        Runs the compilation loop (sequentially adds layers of quantum_circuits until the error is lower than the given threshold).
        """

        state_pointer = 0
        t_array = np.zeros(self.max_n_iter)

        for i_it in range(self.max_n_iter):
            state_pointer = self.apply_local_gate(state_pointer)
            # print(self.next_state)
            # print(np.count_nonzero(self.next_state))
            self.next_state[state_pointer] = 1
            state_pointer += 1
            err, opt_angles, dt = self.optimize_step()
            print(err)
            print(self.next_state)
            t_array[i_it] = dt

            self.errors.append(err)
            num_gates = np.count_nonzero(self.next_state)
            self.number_of_gates.append(num_gates)
            self.angles.append(opt_angles)

            seq = []
            for i_gate in range(num_gates):
                gate_type_number = int(self.next_state[i_gate]) - 1
                seq.append(self.gate_names[gate_type_number])

            self.gate_sequences.append(seq)

            if err < 1e-2 or state_pointer >= self.max_n_gates - 3:
                np.savetxt("time_array_layercomp.txt", t_array)
                break

        return

    def optimize_circuit(self, gate_pos_array):
        self.next_state = gate_pos_array
        num_gates, num_angles, gate_sequence = self.angle_count()
        print(num_gates, num_angles, gate_sequence)

        if self.library == "numba":
            num_params = num_angles
        else:
            num_params = 2 * num_gates

        self.start_values = 2 * np.pi * np.random.randn(self.n_shots, num_params)
        err, opt_angles, dt = self.minimize_cost_(self.cost, num_gates, gate_pos_array, self.start_values)
        print(opt_angles.shape)
        if self.library == "numba":
            _, _, _, U, _ = self.cs_to_unitaries(gate_pos_array, opt_angles, num_gates, self.U0)
        else:
            U, _ = self.cs_to_unitaries(gate_pos_array, opt_angles, self.U0)
        print(U)
        return err, opt_angles, dt, U

    def optimize_step(self):

        """
        Places a new layer of quantum_circuits on the circuit and optimizes it.
        :return: Tuple with the best angles and the best cost.
        """

        num_gates, num_angles, gate_sequence = self.angle_count()
        if self.library == "numba":
            num_params = num_angles
        else:
            num_params = 2 * num_gates

        self.start_values = 2 * np.pi * np.random.randn(self.n_shots, num_params)
        err, opt_angles, dt = self.minimize_cost_(self.cost, num_gates, self.next_state, self.start_values)

        return err, opt_angles, dt
