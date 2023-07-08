import numpy as np
from numba import njit
from quantum_circuits.gate_set_numba import Z_dot_U, U_dot_Z, tensor
from typing import Callable, Tuple, List, Union


@njit(fastmath=True, nogil=True)
def state_fidelity(state: np.ndarray, target_state: np.ndarray) -> float:

    """
    Calculates the fidelity between two states.
    :param state: The state to be compared to the target state.
    :param target_state: The target state.
    :return: The fidelity between the two states.
    """

    overlap: np.ndarray = state.T.conjugate().dot(target_state)[0, 0]
    return np.real(overlap * overlap.conjugate())


@njit(fastmath=True, nogil=True)
def diff_state_fidelity(prop_target: np.ndarray, diff_state: np.ndarray, state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:

    """
    Calculates the derivative of the fidelity between two states.
    :param prop_target: Propagated target state.
    :param diff_state: Derivative of the parametrized quantum state.
    :param state: parametrized quantum state.
    :param target_state: target quantum state.
    :return: derivative of the fidelity between two states.
    """

    diff_overlap: np.ndarray = prop_target.dot(diff_state)[0, 0]
    overlap: np.ndarray = state.T.conjugate().dot(target_state)[0, 0]

    out: np.ndarray = overlap * diff_overlap

    return 2 * np.real(out)


@njit(fastmath=True, nogil=True)
def avg_gate_fidelity(unitary: np.ndarray, target_gate: np.ndarray) -> float:

    """
    Calculates the average gate fidelity between two unitaries.
    :param unitary: parametrized unitary of the quantum circuit.
    :param target_gate: target unitary.
    :return: average gate fidelity between two unitaries.
    """

    F_mat: np.ndarray = target_gate.conjugate().transpose().dot(unitary)
    gate_fidelity: np.ndarray = np.absolute(np.trace(F_mat)) ** 2 / (target_gate.shape[0]) ** 2

    return np.absolute(gate_fidelity)


@njit(fastmath=True, nogil=True)
def diff_gate_fidelity(diff_unitary, unitary, target_gate):

    """
    Calculates the derivative of the average gate fidelity between two unitaries.
    :param diff_unitary: Derivative of the parametrized unitary.
    :param unitary: Parametrized unitary.
    :param target_gate: Target unitary.
    :return: Derivative of the average gate fidelity between two unitaries.
    """

    fmat: np.ndarray = unitary.conjugate().transpose().dot(target_gate)
    fmat_diff: np.ndarray = target_gate.conjugate().transpose().dot(diff_unitary)
    out: np.ndarray = np.trace(fmat) * np.trace(fmat_diff) / (target_gate.shape[0]) ** 2

    return 2 * np.real(out)


@njit(fastmath=True, nogil=True)
def recursive_diff_fidelity(diff_target, diff_unitary, unitary, target_gate):

    """
    Recursively calculates the gradient of the fidelity of a unitary with respect to the unitary.
    :param diff_target: Propagated target unitary.
    :param diff_unitary: Derivative of the parametrized unitary.
    :param unitary: Parametrized unitary.
    :param target_gate: Target unitary.
    :return: Derivative of the fidelity of a unitary with respect to the unitary.
    """

    fmat = target_gate.conjugate().transpose().dot(unitary)
    fmat_diff = diff_target.dot(diff_unitary)
    out = np.trace(fmat).conjugate() * np.trace(fmat_diff) / (target_gate.shape[0]) ** 2

    return 2 * np.real(out)


def create_cost_states_standard(num_qubits, target_state, init_state, *gate_funcs):

    """
    Creates a cost function for a circuit with multiple Z quantum_circuits.
    :param structure: The structure of the circuit.
    :param num_qubits: The number of qubits in the circuit.
    :param target_state: The target state.
    :param init_state: The initial state.
    :param gate_funcs: The gate functions.
    :return: A cost function for the circuit.
    """

    ms = gate_funcs[0]
    cxy = gate_funcs[1]
    z = gate_funcs[2]
    current_operation = tensor([np.eye(2, dtype=np.complex128)] * num_qubits)

    @njit(fastmath=True, nogil=True)
    def cs_to_unitaries(circuit_array: np.ndarray, angle_params: np.ndarray, action_position: np.int64):

        """
        Calculates the cost and gradient of the circuit.
        :param circuit_array: Array of integers representing the circuit.
        :param angle_params: The angles of the circuit.
        :param action_position: Actual number of gates on the circuit
        :return: The cost and gradient of the circuit.
        """

        angle_count = 0
        left_states_arr = np.zeros((init_state.shape[1], init_state.shape[0], action_position + 1),
                                   dtype=np.complex128)
        grad_unitaries_arr = np.zeros((action_position + 1, 2, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)
        grd_idx_array = np.zeros((action_position + 1), np.int8)
        gate_unitaries_arr = np.zeros((action_position + 1, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)
        left_state = target_state.T.conjugate()
        left_states_arr[:, :, 0] = left_state
        # print(action_position)

        current_state = init_state

        for i_ns in range(action_position):
            gate_type_number = int(circuit_array[i_ns]) - 1

            if gate_type_number == 0:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = ms(theta, phi)
                grd_idx_array[i_ns] = 2

                grad_unitaries_arr[i_ns, :, :, :] = \
                    np.concatenate((gate_grad_theta, gate_grad_phi)).reshape(2, current_operation.shape[0],
                                                                             current_operation.shape[1])
            elif gate_type_number == 1:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = cxy(theta, phi)
                grd_idx_array[i_ns] = 2
                grad_unitaries_arr[i_ns, :, :, :] = \
                    np.concatenate((gate_grad_theta, gate_grad_phi)).reshape(2, current_operation.shape[0],
                                                                             current_operation.shape[1])

            else:
                delta = angle_params[angle_count]
                angle_count += 1
                # print(gate_type_number)
                gate, grad_delta = z(delta, gate_type_number - 2)
                gate = np.diag(gate)
                grd_idx_array[i_ns] = 1
                grad_unitaries_arr[i_ns, 0, :, :] = np.diag(grad_delta)
            # print(i_ns, time.time() - start_time)
            gate = np.asfortranarray(gate)
            gate_unitaries_arr[i_ns] = gate
            left_state = left_state.dot(gate)
            left_states_arr[:, :, i_ns + 1] = left_state

        for i_ns in range(action_position):
            current_state = gate_unitaries_arr[action_position - i_ns - 1].dot(current_state)

        return gate_unitaries_arr, grad_unitaries_arr, left_states_arr, current_state, grd_idx_array

    @njit(fastmath=True, nogil=True)
    def cost_and_grad(next_state, angles: np.ndarray, action_position: np.int32):

        """
        Calculates the cost and gradient of the circuit.
        :param next_state: Array of integers representing the circuit.
        :param angles: The angles of the circuit.
        :param action_position: Actual number of gates on the circuit
        :return: The cost and gradient of the circuit.
        """

        cost_grad = np.zeros_like(angles, dtype=np.float64)
        angle_count = 0
        gate_operations_arr, gate_gradients_arr, left_states, current_state, grd_idx_array \
            = cs_to_unitaries(next_state, angles, action_position)

        right_state = current_state

        for i_ap in range(action_position):
            right_state = gate_operations_arr[i_ap, :, :].conjugate().transpose().dot(
                right_state)

            for i_grd in range(grd_idx_array[i_ap]):
                grd_matrix = gate_gradients_arr[i_ap, i_grd, :, :]
                grad_state = left_states[:, :, i_ap].dot(grd_matrix)

                cost_grad[angle_count] = diff_state_fidelity(grad_state,
                                                             right_state,
                                                             current_state,
                                                             target_state)

                angle_count += 1

        err = 1 - state_fidelity(current_state, target_state)

        return err, -np.real(cost_grad)

    return cs_to_unitaries, cost_and_grad


def create_cost_gates_standard(target_gate, *gate_funcs):

    """
    Creates a cost function for a circuit with multiple Z quantum_circuits.
    :param num_qubits:  Number of qubits in the circuit.
    :param target_gate: The target gate.
    :param gate_funcs:  The functions that create the gates.
    :param structure:  True if compilation happens to be in layers, False otherwise.
    :return: Function mapping circuit representation and angles to unitary, Cost function and gradient.
    """

    ms = gate_funcs[0]
    cxy = gate_funcs[1]
    z = gate_funcs[2]

    @njit(fastmath=True, nogil=True)
    def cs_to_unitaries(circuit_array: np.ndarray, angle_params: np.ndarray, action_position: np.int64,
                        current_operation: np.ndarray):
        """
        Calculates the unitaries of the circuit.
        :param circuit_array: Array of integers representing the circuit.
        :param angle_params: The angles of the circuit.
        :param action_position: Actual number of gates on the circuit.
        :param current_operation: The current operation.
        :return: The unitaries of the circuit.
        """
        #print(ms, cxy, z)
        angle_count = 0

        left_unitary = current_operation
        left_unitaries_arr = np.zeros((action_position + 1, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)
        gate_unitaries_arr = np.zeros((action_position + 1, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)
        grad_unitaries_arr = np.zeros((action_position + 1, 2, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)

        grd_idx_array = np.zeros((action_position + 1), np.int8)
        left_unitaries_arr[0, :, :] = left_unitary
        # print(action_position)
        for i_ns in range(action_position):
            gate_type_number = int(circuit_array[i_ns]) - 1

            if gate_type_number == 0:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                # print(ms)
                # print(theta, phi)
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = ms(theta, phi)
                # print("Numba", gate)
                grd_idx_array[i_ns] = 2

                grad_unitaries_arr[i_ns, :, :, :] = \
                    np.concatenate((gate_grad_theta, gate_grad_phi)).reshape(2, current_operation.shape[0],
                                                                             current_operation.shape[1])
                gate_unitaries_arr[i_ns, ::] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)

            elif gate_type_number == 1:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = cxy(theta, phi)
                grd_idx_array[i_ns] = 2
                grad_unitaries_arr[i_ns, :, :, :] = \
                    np.asfortranarray(
                        np.concatenate((gate_grad_theta, gate_grad_phi)).reshape(2, current_operation.shape[0],
                                                                                 current_operation.shape[1]))
                gate_unitaries_arr[i_ns, ::] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)
            else:
                delta = angle_params[angle_count]
                angle_count += 1
                #print(z)
                gate, grad_delta = z(delta, gate_type_number - 2)
                grd_idx_array[i_ns] = 1
                grad_unitaries_arr[i_ns, 0, :, 0] = np.asfortranarray(grad_delta)
                gate_unitaries_arr[i_ns, :, 0] = gate.conjugate()
                # print(np.diag(gate), delta)
                current_operation = U_dot_Z(current_operation, gate)
            # print(i_ns, time.time() - start_time)

            left_unitaries_arr[i_ns + 1, ::] = np.asfortranarray(current_operation)

        return gate_unitaries_arr, grad_unitaries_arr, left_unitaries_arr, current_operation, grd_idx_array

    @njit(fastmath=True, nogil=True)
    def cost_and_grad(next_state, angles: np.ndarray, action_position: np.int32, starting_u: np.ndarray):

        """
        Calculates the cost and gradient of the circuit.
        :param next_state: Array of integers representing the circuit.
        :param angles:  The angles of the circuit.
        :param action_position: Actual number of gates on the circuit.
        :param starting_u: The starting unitary.
        :return: The cost and gradient of the circuit.
        """

        cost_grad = np.zeros_like(angles, dtype=np.float64)
        angle_count = 0
        gate_unitaries_arr, gate_gradients_arr, left_unitary, current_operation, grd_idx_array \
            = cs_to_unitaries(next_state, angles, action_position, starting_u)
        # left_unitary, current_operation = compute_grape_unitaries(gate_operations_list)
        # print(current_operation.dtype)
        right_unitary = current_operation
        adj_target = target_gate.T.conjugate()
        overlap = np.trace(adj_target.dot(current_operation))
        d = target_gate.shape[0]
        err = 1 - (np.absolute(overlap) / d) ** 2

        for i_ap in range(action_position):
            if grd_idx_array[i_ap] == 1:
                current_gate = np.ascontiguousarray(gate_unitaries_arr[i_ap, :, 0])
                right_unitary = Z_dot_U(current_gate, right_unitary)
            else:
                current_gate = np.ascontiguousarray(gate_unitaries_arr[i_ap, :, :])
                right_unitary = np.dot(current_gate, right_unitary)

            for i_grd in range(grd_idx_array[i_ap]):
                if grd_idx_array[i_ap] == 1:
                    grd_matrix = gate_gradients_arr[i_ap, 0, :, 0]
                    grad_unitary = U_dot_Z(np.dot(adj_target, left_unitary[i_ap, ::]), grd_matrix)
                else:
                    grd_matrix = gate_gradients_arr[i_ap, i_grd, ::]
                    grad_unitary = np.dot(np.dot(adj_target, left_unitary[i_ap, ::]), grd_matrix)
                # print(left_unitary[:, :, i_ap])
                grad_overlap = np.trace(grad_unitary.dot(right_unitary))
                cost_grad[angle_count] = 2 * np.real(grad_overlap * overlap.conjugate() / d ** 2)
                angle_count += 1

        return err, -np.real(cost_grad)

    return cs_to_unitaries, cost_and_grad


def main():
    import timeit
    from gate_set_numba import create_fast_ms_set_numba
    from line_profiler import LineProfiler

    np.random.seed(0)
    num_qubits = 8
    gg, gnames, structure = create_fast_ms_set_numba(num_qubits, z_prod=True)
    #print(gg)
    gate_set_tuple = tuple(gg)
    circuit_state = np.random.randint(1, 3, size=(50,))
    angles_stacked = np.random.randn(30, 5)

    two_angle_gates = ["MS", "Cxy"]
    circuit_state = np.random.randint(1, 4, size=(50,))
    # print(circuit_state)
    angles_init = np.random.randn(200)
    angles_stacked = np.random.randn(50)
    init_gate = np.identity(2 ** num_qubits, dtype=np.complex128)
    print(angles_stacked.dtype)
    target_gate = np.identity(2 ** num_qubits, dtype=np.complex128)
    timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
    timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
    cs, cg = create_cost_gates_standard(num_qubits, target_gate, structure, *gate_set_tuple)

    # print(vec_circuit(circuit_state, angles_init, len(circuit_state), init_gate))
    # print(cs_t_us(circuit_state, angles_init, len(circuit_state), init_gate))

    def test():
        return cg(circuit_state, angles_init, len(circuit_state), init_gate)

    print(test())
    lp = LineProfiler()
    lp.add_function(cg)
    # lp.add_function(cs_t_us)
    lp.add_function(recursive_diff_fidelity)
    lp.add_function(avg_gate_fidelity)
    lp_wrapper = lp(test)
    lp_wrapper()
    lp.print_stats()

    #print(timeit.timeit(test, number=10) / 10)


if __name__ == "__main__":
    main()
