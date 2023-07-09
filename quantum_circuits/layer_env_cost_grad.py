import numpy as np
from numba import njit
from quantum_circuits.gate_set_numba import Z_dot_U, U_dot_Z, tensor
from typing import Callable, Tuple, List, Union


def create_cost_gates_layers(num_qubits, target_gate, *gate_funcs):

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

    #@njit(fastmath=True, nogil=True)
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
        grad_unitaries_arr = np.zeros((action_position + 1, num_qubits, current_operation.shape[0], current_operation.shape[1]),
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

                grad_unitaries_arr[i_ns, :2, :, :] = \
                    np.concatenate((gate_grad_theta, gate_grad_phi)).reshape(2, current_operation.shape[0],
                                                                             current_operation.shape[1])
                gate_unitaries_arr[i_ns, ::] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)

            elif gate_type_number == 1:
                theta, phi = angle_params[angle_count], angle_params[angle_count + 1]
                angle_count += 2
                gate, gate_grad_phi, gate_grad_theta = cxy(theta, phi)
                grd_idx_array[i_ns] = 2
                grad_unitaries_arr[i_ns, :2, :, :] = \
                    np.asfortranarray(
                        np.concatenate((gate_grad_theta, gate_grad_phi)).reshape(2, current_operation.shape[0],
                                                                                 current_operation.shape[1]))
                gate_unitaries_arr[i_ns, ::] = gate.conjugate().transpose()
                current_operation = np.dot(current_operation, gate)
            else:
                delta = angle_params[angle_count: angle_count + num_qubits]
                angle_count += num_qubits
                #print(z)
                gate, grad_delta = z(delta)
                grd_idx_array[i_ns] = num_qubits
                grad_unitaries_arr[i_ns, :, :, 0] = np.asfortranarray(grad_delta)
                gate_unitaries_arr[i_ns, :, 0] = gate.conjugate()
                # print(np.diag(gate), delta)
                current_operation = U_dot_Z(current_operation, gate)
            # print(i_ns, time.time() - start_time)

            left_unitaries_arr[i_ns + 1, ::] = np.asfortranarray(current_operation)

        return gate_unitaries_arr, grad_unitaries_arr, left_unitaries_arr, current_operation, grd_idx_array

    #@njit(fastmath=True, nogil=True)
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
            if next_state[i_ap] == 3:
                current_gate = np.ascontiguousarray(gate_unitaries_arr[i_ap, :, 0])
                right_unitary = Z_dot_U(current_gate, right_unitary)
            else:
                current_gate = np.ascontiguousarray(gate_unitaries_arr[i_ap, :, :])
                right_unitary = np.dot(current_gate, right_unitary)

            for i_grd in range(grd_idx_array[i_ap]):
                if next_state[i_ap] == 3:
                    grd_matrix = gate_gradients_arr[i_ap, i_grd, :, 0]
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