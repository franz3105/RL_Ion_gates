import numpy as np
import scipy as sp
import scipy.special
from circuit_search.compilation_in_layers import LayerCompilation
import itertools
from operator import itemgetter


def guessables(set, num):
    guesses = []
    for p in itertools.combinations(set, num):
        guesses.append(np.array(p, np.int64))

    return guesses


# This performs an exhaustive search over all possible subcircuits of a given length.
# It extends the LayerCompilation class. It is a brute force search, so it is not very scalable (factorial complexity).
# But it guarantees to find the optimal circuit for small numbers of layers.

class RandomSearch(LayerCompilation):

    def __init__(self, num_layers, num_episodes, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.max_guesses = num_episodes
        self.circuit_list = []
        self.error_list = []
        self.best_circuit_list = []
        self.best_error_list = []
        self.circuit_size_list = []
        self.best_circuit_size_list = []
        self.best_angles = []

    def run_compilation(self):
        state_pointer = 0

        for i_it in range(self.max_n_iter):
            state_pointer = self.apply_local_gate(state_pointer)
            # print(self.next_state)
            # print(np.count_nonzero(self.next_state))
            self.next_state[state_pointer] = 1
            state_pointer += 1
            err, opt_angles = self.optimize()

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
                break

        return

    def angle_count_circ(self, circuit):

        num_gates = np.count_nonzero(circuit)
        gate_sequence = []
        num_angles = 0

        for i_gate in range(num_gates):
            gate_type_number = int(circuit[i_gate]) - 1
            gate_sequence.append(self.gate_names[gate_type_number])
            if self.gate_names[gate_type_number] in self.two_angle_gates:
                num_angles += 2
            elif self.gate_names[gate_type_number] in ["R_xyz_1", "R_xyz_2", "R_xyz_3"]:
                num_angles += 3
            else:
                num_angles += 1

        return num_gates, num_angles, gate_sequence

    def num_combinations(self, start):

        l_comb = 1
        ms_positions = np.arange(1, self.num_layers + 1) * (self.num_qubits + 3) - 1

        for l_idx in range(start, self.num_layers + 1):
            for m in range(1, (l_idx + 1) * self.num_qubits + 3 * (l_idx + 1) - 1):
                comb_positions = filter(lambda a: a not in ms_positions,
                                        range((l_idx + 1) * self.num_qubits + 3 * (l_idx + 1) - 1))
                l_comb += len(guessables(comb_positions, m))

        return l_comb

    def run_search(self):

        checked_combinations = []
        start = 1
        num_comb = self.num_combinations(start)
        layer_circuit = np.zeros((self.num_layers + 1, self.num_qubits + 2))
        s_idx = 0

        for l_idx in range(start, self.num_layers + 1):
            print(l_idx)
            layer_circuit[l_idx, 0] = 2
            for i_z in range(self.num_qubits):
                layer_circuit[l_idx, i_z + 1] = 3 + i_z

            layer_circuit[l_idx, self.num_qubits + 1] = 2
            ms_positions = np.arange(1, self.num_layers + 1) * (self.num_qubits + 3) - 1

            general_circuit = []
            general_circuit += list(layer_circuit[l_idx, :])

            for i_l in range(l_idx):
                print(l_idx)
                general_circuit += [1]
                general_circuit += list(layer_circuit[l_idx, :])

            general_circuit = np.array(general_circuit)

            num_gates, _, self.gate_sequence = self.angle_count_circ(general_circuit)

            for m in range(1, (l_idx + 1) * self.num_qubits + 3 * (l_idx + 1) - 1):
                comb_positions = filter(lambda a: a not in ms_positions,
                                        range((l_idx + 1) * self.num_qubits + 3 * (l_idx + 1) - 1))
                # print(comb_positions)
                all_subcirc_combinations = guessables(comb_positions, m)
                # print(all_subcirc_combinations)

                for i_guess in range(self.max_guesses):
                    circuit = general_circuit.copy()
                    idx = np.random.randint(0, len(all_subcirc_combinations))
                    circuit = np.delete(circuit, all_subcirc_combinations[idx])
                    # print(circuit)
                    checked_combinations.append(circuit)
                    _, _, gate_seq = self.angle_count_circ(circuit)
                    n_gates_el, _, new_seq = self.angle_count_circ(circuit)
                    num_angles = 2 * n_gates_el
                    alpha0 = 2 * np.pi * np.random.randn(self.n_shots, num_angles)
                    err, opt_angles, *_ = self.minimize_cost_(self.cost, len(circuit), circuit, alpha0)
                    print(err)
                    s_idx += 1
                    print(f"{s_idx}/{self.max_guesses}")
                    if err < 1e-2:
                        self.best_circuit_list.append(circuit)
                        self.best_error_list.append(err)
                        self.best_circuit_size_list.append(len(circuit))
                        self.best_angles.append(opt_angles)

        return


def main():
    es = RandomSearch()
    es.run_search()
