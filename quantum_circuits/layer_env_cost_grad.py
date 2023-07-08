import numpy as np
from numba import njit
from quantum_circuits.gate_set_numba import Z_dot_U, U_dot_Z, tensor
from typing import Callable, Tuple, List, Union


def create_cost_gates_layers(num_qubits: int, target_gate: np.ndarray,
                             *gate_funcs: Union[Callable, List[Callable]]) -> Callable:
    """
    Creates a cost function for a single Z gate.
    :param num_qubits: Number of qubits in the circuit.
    :param target_gate: Target unitary.
    :param gate_funcs: List of gate functions.
    :return: Cost function for a single Z gate.
    """

    ms: Callable = gate_funcs[0]
    cxy: Callable = gate_funcs[1]
    z: Callable = gate_funcs[2]

    @njit(fastmath=True, nogil=True)
    def cs_to_unitaries(circuit_array: np.ndarray, angle_params: np.ndarray, action_position: np.int64,
                        current_operation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Calculates the unitary of a circuit.
        :param circuit_array: Array of integers representing which gate is at which position in the circuit.
        Empty positions are represented by 0.
        :param angle_params: Array of angles for the parametrized quantum_circuits.
        :param action_position: Index of the last gate on the circuit.
        :param current_operation: Starting unitary which is multiplied with the unitaries on the circuit.
        :return: Unitary of the circuit.
        """

        angle_count = 0
        left_unitary = current_operation
        left_unitaries_arr = np.zeros((action_position + 1, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)
        gate_unitaries_arr = np.zeros((action_position + 1, current_operation.shape[0], current_operation.shape[1]),
                                      dtype=np.complex128)
        grad_unitaries_arr = np.zeros((angle_params.shape[0], current_operation.shape[0],
                                       current_operation.shape[1]),
                                      dtype=np.complex128)
        grd_idx_array = np.zeros((action_position + 1), np.int8)
        left_unitaries_arr[0, :, :] = left_unitary
        # print(action_position)
        for i_ns in range(action_position):
            gate_type_number = int(circuit_array[i_ns]) - 1

            if gate_type_number == 0:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]

                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = ms(theta, phi)
                grd_idx_array[i_ns] = 2

                grad_unitaries_arr[angle_count - 2, ::] = gate_grad_theta
                grad_unitaries_arr[angle_count - 1, ::] = gate_grad_phi
                gate_unitaries_arr[i_ns, :, :] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)

            elif gate_type_number == 1:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = cxy(theta, phi)
                grd_idx_array[i_ns] = 2

                grad_unitaries_arr[angle_count - 2, ::] = gate_grad_theta
                grad_unitaries_arr[angle_count - 1, ::] = gate_grad_phi

                gate_unitaries_arr[i_ns, :, :] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)
            else:
                delta = angle_params[angle_count:angle_count + num_qubits]
                angle_count += num_qubits
                gate, grad_delta = z(delta)
                grd_idx_array[i_ns] = num_qubits
                grad_unitaries_arr[angle_count - num_qubits:angle_count, :, 0] = grad_delta
                gate_unitaries_arr[i_ns, :, 0] = gate.conjugate()
                current_operation = U_dot_Z(current_operation, gate)

            left_unitaries_arr[i_ns + 1, :, :] = current_operation

        return gate_unitaries_arr, grad_unitaries_arr, left_unitaries_arr, current_operation, grd_idx_array

    @njit(fastmath=True, nogil=True)
    def cost_and_grad(next_state, angles: np.ndarray, action_position: np.int32, starting_u: np.ndarray):

        """
        Calculates the cost and gradient of a circuit.
        :param next_state: Array of integers representing which gate is at which position in the circuit.
        :param angles: Array of angles for the parametrized quantum_circuits.
        :param action_position: Index of the last gate on the circuit.
        :param starting_u: Starting unitary which is multiplied with the unitaries on the circuit.
        :return: Cost and gradient of the circuit unitary.
        """

        cost_grad = np.zeros_like(angles, dtype=np.float64)
        angle_count = 0
        gate_unitaries_arr, gate_gradients_list, left_unitary, current_operation, grd_idx_array \
            = cs_to_unitaries(next_state, angles, action_position, starting_u)

        right_unitary = current_operation
        adj_target = target_gate.T.conjugate()
        print(current_operation)
        overlap = np.trace(adj_target.dot(current_operation))
        d = target_gate.shape[0] ** 2
        err = 1 - (np.abs(overlap) / d) ** 2

        for i_ap in range(action_position):
            grd_index = np.sum(grd_idx_array[:i_ap])

            if grd_idx_array[i_ap] == num_qubits:
                current_gate = gate_unitaries_arr[i_ap, :, 0]
                right_unitary = Z_dot_U(current_gate, right_unitary)
            else:
                current_gate = gate_unitaries_arr[i_ap, :, :]
                right_unitary = np.dot(current_gate, right_unitary)

            for i_grd in range(grd_idx_array[i_ap]):
                if grd_idx_array[i_ap] == num_qubits:
                    grd_matrix = gate_gradients_list[grd_index + i_grd, :, 0]
                    grad_unitary = U_dot_Z(np.dot(adj_target, left_unitary[i_ap, :, :]), grd_matrix)
                else:
                    grd_matrix = gate_gradients_list[grd_index + i_grd, ::]
                    grad_unitary = np.dot(np.dot(adj_target, left_unitary[i_ap, :, :]), grd_matrix)
                # print(left_unitary[:, :, i_ap])
                grad_overlap = np.trace(grad_unitary.dot(right_unitary))
                cost_grad[angle_count] = 2 * np.real(grad_overlap * overlap.conjugate() / d ** 2)
                angle_count += 1

        return err, -np.real(cost_grad)

    return cs_to_unitaries, cost_and_grad


def create_cost_states_layers(num_qubits: int, target_gate: np.ndarray,
                              *gate_funcs: Union[Callable, List[Callable]]) -> Callable:
    """
    Creates a cost function for a single Z gate.
    :param num_qubits: Number of qubits in the circuit.
    :param target_gate: Target unitary.
    :param gate_funcs: List of gate functions.
    :return: Cost function for a single Z gate.
    """

    ms: Callable = gate_funcs[0]
    cxy: Callable = gate_funcs[1]
    z: Callable = gate_funcs[2]

    @njit(fastmath=True, nogil=True)
    def cs_to_unitaries_states(circuit_array: np.ndarray, angle_params: np.ndarray, action_position: np.int64,
                               current_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Calculates the unitary of a circuit.
        :param circuit_array: Array of integers representing which gate is at which position in the circuit.
        Empty positions are represented by 0.
        :param angle_params: Array of angles for the parametrized quantum_circuits.
        :param action_position: Index of the last gate on the circuit.
        :param current_state: Starting unitary which is multiplied with the unitaries on the circuit.
        :return: Unitary of the circuit.
        """

        angle_count = 0
        left_unitary = current_state
        left_unitaries_arr = np.zeros((action_position + 1, current_state.shape[0]),
                                      dtype=np.complex128)
        gate_unitaries_arr = np.zeros((action_position + 1, current_state.shape[0]),
                                      dtype=np.complex128)
        grad_unitaries_arr = np.zeros((angle_params.shape[0], current_state.shape[0]),
                                      dtype=np.complex128)
        grd_idx_array = np.zeros((action_position + 1), np.int8)
        left_unitaries_arr[0, :, :] = left_unitary
        # print(action_position)
        for i_ns in range(action_position):
            gate_type_number = int(circuit_array[i_ns]) - 1

            if gate_type_number == 0:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]

                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = ms(theta, phi)
                grd_idx_array[i_ns] = 2

                grad_unitaries_arr[angle_count - 2, ::] = gate_grad_theta
                grad_unitaries_arr[angle_count - 1, ::] = gate_grad_phi
                gate_unitaries_arr[i_ns, :, :] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)

            elif gate_type_number == 1:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = cxy(theta, phi)
                grd_idx_array[i_ns] = 2

                grad_unitaries_arr[angle_count - 2, ::] = gate_grad_theta
                grad_unitaries_arr[angle_count - 1, ::] = gate_grad_phi

                gate_unitaries_arr[i_ns, :, :] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)
            else:
                delta = angle_params[angle_count:angle_count + num_qubits]
                angle_count += num_qubits
                gate, grad_delta = z(delta)
                grd_idx_array[i_ns] = num_qubits
                grad_unitaries_arr[angle_count - num_qubits:angle_count, :, 0] = grad_delta
                gate_unitaries_arr[i_ns, :, 0] = gate.conjugate()
                current_operation = U_dot_Z(current_operation, gate)

            left_unitaries_arr[i_ns + 1, :, :] = current_operation

        return gate_unitaries_arr, grad_unitaries_arr, left_unitaries_arr, current_operation, grd_idx_array

    @njit(fastmath=True, nogil=True)
    def cost_and_grad(next_state, angles: np.ndarray, action_position: np.int32, starting_u: np.ndarray):

        """
        Calculates the cost and gradient of a circuit.
        :param next_state: Array of integers representing which gate is at which position in the circuit.
        :param angles: Array of angles for the parametrized quantum_circuits.
        :param action_position: Index of the last gate on the circuit.
        :param starting_u: Starting unitary which is multiplied with the unitaries on the circuit.
        :return:
        """

        cost_grad = np.zeros_like(angles, dtype=np.float64)
        angle_count = 0
        gate_unitaries_arr, gate_gradients_list, left_unitary, current_operation, grd_idx_array \
            = cs_to_unitaries_states(next_state, angles, action_position, starting_u)

        right_unitary = current_operation
        adj_target = target_gate.T.conjugate()
        print(current_operation)
        overlap = np.trace(adj_target.dot(current_operation))
        d = target_gate.shape[0] ** 2
        err = 1 - (np.abs(overlap) / d) ** 2

        for i_ap in range(action_position):
            grd_index = np.sum(grd_idx_array[:i_ap])

            if grd_idx_array[i_ap] == num_qubits:
                current_gate = gate_unitaries_arr[i_ap, :, 0]
                right_unitary = Z_dot_U(current_gate, right_unitary)
            else:
                current_gate = gate_unitaries_arr[i_ap, :, :]
                right_unitary = np.dot(current_gate, right_unitary)

            for i_grd in range(grd_idx_array[i_ap]):
                if grd_idx_array[i_ap] == num_qubits:
                    grd_matrix = gate_gradients_list[grd_index + i_grd, :, 0]
                    grad_unitary = U_dot_Z(np.dot(adj_target, left_unitary[i_ap, :, :]), grd_matrix)
                else:
                    grd_matrix = gate_gradients_list[grd_index + i_grd, ::]
                    grad_unitary = np.dot(np.dot(adj_target, left_unitary[i_ap, :, :]), grd_matrix)
                # print(left_unitary[:, :, i_ap])
                grad_overlap = np.trace(grad_unitary.dot(right_unitary))
                cost_grad[angle_count] = 2 * np.real(grad_overlap * overlap.conjugate() / d ** 2)
                angle_count += 1

        return err, -np.real(cost_grad)

    return cs_to_unitaries_states, cost_and_grad
