import jax
import jax.numpy as jnp
import jaxopt
import os
import numpy as np
from jax import jit, lax, vmap, value_and_grad, pmap
from functools import partial
from typing import List, Callable, Tuple

@jit
def tensor(op_list: List[jnp.ndarray]) -> jnp.ndarray:
    """
    op_list: list of 2d arrays
    :return: tensor product of all matrices in op_list
    """

    s = op_list[0]

    for i_op in range(1, len(op_list)):
        s = jnp.kron(s, op_list[i_op])

    return s


@jit
def state_fidelity(state: jnp.ndarray, target_state: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the fidelity between two states.
    :param state: jnp.ndarray
                Quantum state.
    :param target_state: jnp.ndarray
            Target quantum state (jnp.ndarray).
    :return:

    Fidelity between the two states (jnp.ndarray).
    """

    overlap = state.T.conjugate().dot(target_state)[0, 0]
    return jnp.real(overlap * overlap.conjugate())


@jit
def diff_state_fidelity(diff_target: jnp.ndarray, diff_state: jnp.ndarray, state: jnp.ndarray,
                        target_state: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the derivative of the fidelity between two states.
    :param diff_target: Derivative of the target state.
    :param diff_state: Derivative of the state.
    :param state: State.
    :param target_state: Target state.
    :return:
    """

    diff_overlap = diff_target.dot(diff_state)[0, 0]
    overlap = state.T.conjugate().dot(target_state)[0, 0]

    out = overlap * diff_overlap

    return 2 * jnp.real(out)


@jit
def avg_gate_fidelity(unitary: jnp.ndarray, target_gate: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the average gate fidelity between two unitaries.
    :param unitary: Unitary.
    :param target_gate: Target unitary.
    :return:
    """

    F_mat = target_gate.conjugate().transpose().dot(unitary)
    gate_fidelity = jnp.absolute(jnp.trace(F_mat)) ** 2 / (target_gate.shape[0]) ** 2

    return jnp.absolute(gate_fidelity)


def diff_gate_fidelity(diff_overlap: jnp.ndarray, overlap: jnp.ndarray, d: int):
    """
    Calculates the derivative of the average gate fidelity between two unitaries.
    :param diff_overlap: Derivative of the overlap.
    :param overlap: Overlap.
    :param d: Dimension of the unitaries.
    :return:
    """

    out = overlap * jnp.trace(diff_overlap) / d ** 2

    return 2 * jnp.real(out)


@jit
def recursive_diff_fidelity(diff_target: jnp.ndarray, diff_unitary: jnp.ndarray, unitary: jnp.ndarray,
                            target_gate: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the derivative of the average gate fidelity between two unitaries.
    :param diff_target: Derivative of the target unitary.
    :param diff_unitary: Derivative of the unitary.
    :param unitary: Unitary.
    :param target_gate: Target unitary.
    :return:
    """

    fmat = unitary.conjugate().transpose().dot(target_gate)
    fmat_diff = diff_target.dot(diff_unitary)

    out = jnp.trace(fmat) * jnp.trace(fmat_diff) / (target_gate.shape[0]) ** 2

    return 2 * jnp.real(out)


@jit
def special_prod(index: int, u_1: jnp.ndarray, u_2: jnp.ndarray, u_3: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the product of three unitaries.
    :param index: Index of the unitary to be calculated (int).
    :param u_1: First unitary (jnp.ndarray).
    :param u_2: Second unitary (jnp.ndarray).
    :param u_3: Third unitary (jnp.ndarray).
    :return:
    """

    return lax.cond(jnp.less_equal(index - 1, 2), lambda a, b, c: a * c, lambda a, b, c: jnp.dot(a, b), u_1, u_2, u_3)


@jit
def special_append(index, y1, y2, y3):
    """
    Appends the result of a special product to a list.
    :param index: Index of the unitary to be calculated (int).
    :param y1: First unitary (jnp.ndarray).
    :param y2: Second unitary (jnp.ndarray).
    :param y3: Third unitary (jnp.ndarray).
    :return: List of unitaries (jnp.ndarray).
    """

    return lax.cond(jnp.less_equal(index - 1, 2), lambda a, b, c: a.append(c), lambda a, b, c: b.append(c), y1, y2, y3)


@jit
def check_index(index):
    """
    Checks if the index is less than 2.
    :param index:
    :return:
    """

    return lax.cond(jnp.less_equal(index - 1, 2), lambda _: 0, lambda _: 1, index)


@jit
def cum_special_prod(pos_array, two_angles_gates, z_gates, U_init):
    """
    Calculates the cumulative product of unitaries.
    :param pos_array: Array determining the position of the unitaries (jnp.ndarray).
    :param two_angles_gates: List of unitaries (jnp.ndarray).
    :param z_gates: List of unitaries (jnp.ndarray).
    :param U_init: Initial unitary (jnp.ndarray).
    :return: List of unitaries (jnp.ndarray).
    """

    @jit
    def lax_special_dot(carry, index_arr):
        index = index_arr[0]
        pos = index_arr[1]
        y1 = special_prod(index, carry, two_angles_gates[pos], z_gates[pos])
        y2 = y1.T.conjugate()
        carry = y1
        return carry, [y1, y2]

    out = lax.scan(lax_special_dot, U_init, pos_array)
    current_op = out[0]

    left_unitaries, right_unitaries = jnp.split(jnp.array(out[1]), 2, axis=0)

    return current_op, left_unitaries[0, ::], right_unitaries[0, ::]


@jit
def cumulative_dot(U_array, U_init):
    """
    Calculates the cumulative product of unitaries.
    :param U_array: List of unitaries (jnp.ndarray).
    :param U_init: Initial unitary (jnp.ndarray).
    :return: List of unitaries (jnp.ndarray).
    """

    @jit
    def lax_dot(carry, x):
        y1 = carry.dot(x[0])
        y2 = y1.T.conjugate()
        grd_U_1 = carry.dot(x[1]).dot(y2)
        grd_U_2 = carry.dot(x[2]).dot(y2)
        return y1, [grd_U_1, grd_U_2]

    out = lax.scan(lax_dot, U_init, U_array)
    current_op = out[0]
    dU = jnp.array(out[1])

    return current_op, dU


def create_simple_cost_gates(target_gate, *gate_funcs, n_devices=1):
    """
    Creates the cost function for the quantum_circuits.
    :param n_devices: Number of devices.
    :param target_gate: Target gate.
    :param gate_funcs: List of gate functions.
    :return: List of unitaries (jnp.ndarray).
    """
    d = target_gate.shape[0]

    @jit
    def diff_grad_fidelity(left_us, grad_U, right_us, t_prod, overlap, d):
        """
        Calculates the gradient of the fidelity.
        :param left_us: Left product of unitaries up to grad_U.
        :param grad_U: Gradient of a single unitary.
        :param right_us: Right product of unitaries from grad_U to the end.
        :param t_prod : Product of target unitaries.
        :param overlap: Overlap between V and U, i.e., tr(VU.T.conj()).
        :param d: Dimension of the Hilbert space.
        :return: Gradient of the fidelity.
        """

        return diff_gate_fidelity(left_us.dot(grad_U).dot(right_us).dot(t_prod), overlap, d)

    @jit
    def cs_to_unitaries(circuit_state: jnp.ndarray, angle_params: jnp.ndarray, current_operation):
        """
        Creates the unitaries from the parameters.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params:  Angles of each gate.
        :param current_operation: Starting unitary.
        :return: Cumulative products of the unitaries on the circuit, Gradients of the unitaries.
        """

        angle_params = angle_params.reshape(len(circuit_state), 2, order="F")
        # print(f"Angles", angle_params)

        result_circuit_2angles = pmap(lambda c, a: vmap(lambda i, x:
                                                        lax.switch(i - 1, *gate_funcs, x))(c, a))(
            circuit_state.reshape(n_devices, -1), angle_params.reshape(n_devices, -1, 2))

        # result_circuit_2angles = vmap(lambda i, x:lax.switch(i - 1, *gate_funcs, x))(circuit_state, angle_params)
        gates_array, grad_theta_array, grad_phi_array = map(lambda c: c.reshape(-1, *(d, d)), result_circuit_2angles)
        # gates_array, grad_theta_array, grad_phi_array = result_circuit_2angles

        current_operation, dU = cumulative_dot(
            jnp.stack([gates_array, grad_theta_array, grad_phi_array]).swapaxes(0, 1),
            current_operation)

        return current_operation, dU

    @jit
    def cost_and_grad(circuit_state, angle_params, action_position, current_operation):
        """
        Calculates the cost and the gradient.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params: Angles of each gate.
        :param action_position: Position of the last gate.
        :param current_operation: Starting unitary.
        :return: Cost and gradient.
        """

        current_operation, dU = \
            cs_to_unitaries(circuit_state, angle_params, current_operation)
        # print(current_operation)
        t_prod = current_operation.dot(target_gate.T.conjugate())

        overlap = jnp.trace(t_prod).conjugate()
        err = 1 - jnp.absolute(overlap) ** 2 / d ** 2

        # print(right_unitaries_arr.shape)
        # cost_grad = vmap(lambda y: vmap(
        #    lambda a: (2 / d ** 2) * jnp.real(overlap * jnp.trace(a.dot(t_prod))))(y))(
        #    dU)

        cost_grad = pmap(lambda y:
                         vmap(lambda a: (2 / d ** 2) *
                                        jnp.real(overlap * jnp.trace(a.dot(t_prod))))(y))(
            dU.reshape(n_devices, -1, *(d, d)))

        return err, -cost_grad.ravel()

    @jit
    def vectorized_cost(circuit_state, angle_params, action_position, current_operation):
        """
        Calculates the cost.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params: Angles of each gate.
        :param action_position: Position of the last gate.
        :param current_operation: Starting unitary.
        :return: Cost, vectorized for multiple arrays of input parameters.
        """

        all_err, all_grads = jax.vmap(lambda a: cost_and_grad(circuit_state, a, action_position,
                                                              current_operation))(angle_params)
        # print(all_err)
        return jnp.min(all_err), all_grads

    return cs_to_unitaries, cost_and_grad, vectorized_cost


def pvec_map():
    """
    Function used to map a function to multiple arrays of input parameters depending wether gpus or cpus are used.
    :return:  Function that maps a function to multiple arrays of input parameters.
    """

    n_devices = jax.device_count("gpu")

    if jax.device_count("gpu") > 1:

        def map_func(f, *arrays, **kwargs):
            reshaped_args = [jnp.asarray(a).reshape(n_devices, -1, *a.shape[1:]) for a in arrays]
            return pmap(lambda *a: vmap(lambda *b: f(*b, **kwargs))(*a))(*reshaped_args)

    else:
        def map_func(f, *arrays, **kwargs):

            return vmap(lambda *a: f(*a, **kwargs))(*arrays)

    return map_func, n_devices


def pvqd_evo_cost(target_gates, *gate_funcs, n_devices=1, n_params_per_gate=2, max_iter=100):
    """
    Creates the cost function for the p-VQD algorithm.
    :param n_params_per_gate: Number of parameters per gate.
    :param target_gates: Target gates.
    :param gate_funcs: List of gate functions.
    :param n_devices: Number of devices.
    :return: List of unitaries (jnp.ndarray).
    """
    d = target_gates[0].shape[0]

    # print(*gate_funcs)
    # print(target_gates)

    @jit
    def diff_grad_fidelity(left_us, grad_U, right_us, t_prod, overlap, d):
        """
        Calculates the gradient of the fidelity.
        :param left_us: Left unitaries.
        :param grad_U: Gradient of the unitary.
        :param right_us: Right unitaries.
        :param t_prod: Product of the target gate and the conjugate of the current gate.
        :param overlap: Overlap of the current gate and the target gate.
        :param d: Dimension of the Hilbert space.
        :return:
        """

        return diff_gate_fidelity(left_us.dot(grad_U).dot(right_us).dot(t_prod), overlap, d)

    @jit
    def cs_to_unitaries(circuit_state: jnp.ndarray, angle_params: jnp.ndarray, current_operation):
        """
        Creates the unitaries from the parameters.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params: Collection of angles of each gate.
        :param current_operation: Starting unitary.
        :return: New unitary, cumulative unitary.
        """
        angle_params = angle_params.reshape(len(circuit_state), n_params_per_gate, order="F")

        result_circuit_2angles = pmap(lambda c, a: vmap(lambda i, x:
                                                        lax.switch(i - 1, *gate_funcs, x))(c, a))(
            circuit_state.reshape(n_devices, -1),
            angle_params.reshape(n_devices, -1, n_params_per_gate))

        gates_array, grad_theta_array, grad_phi_array = map(lambda c: c.reshape(-1, *(d, d)), result_circuit_2angles)

        current_operation, dU = cumulative_dot(
            jnp.stack([gates_array, grad_theta_array, grad_phi_array]).swapaxes(0, 1),
            current_operation)

        return current_operation, dU

    @jit
    def cost_and_grad(tg, circuit_state, angle_params, action_position, current_operation):
        """
        Calculates the cost and the gradient.
        :param tg: Target gate.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params: Collection of angles of each gate.
        :param action_position: Position of the last gate.
        :param current_operation: Starting unitary.
        :return: Cost and gradient.
        """

        # new_state = circuit_state[:action_position]
        assert circuit_state[action_position] == 0

        current_operation, dU = \
            cs_to_unitaries(circuit_state, angle_params, current_operation)

        t_prod = current_operation.dot(tg.T.conjugate())

        overlap = jnp.trace(t_prod).conjugate()
        err = 1 - jnp.absolute(overlap) ** 2 / d ** 2

        # print(right_unitaries_arr.shape)
        cost_grad = pmap(lambda y:
                         vmap(lambda a: (2 / d ** 2) *
                                        jnp.real(overlap * jnp.trace(a.dot(t_prod))))(y))(
            dU.reshape(n_devices, -1, *(d, d)))

        return err, -cost_grad.ravel()

    @jit
    def pvd_cost(circuit_state, angle_params, action_position):
        """
        Calculates the cost.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params: Collection of angles of each gate.
        :param action_position: Position of the last gate.
        :return: Cost, vectorized for multiple arrays of input parameters.
        """

        #print(angle_params)
        all_err, all_grads = jax.vmap(lambda a, b: cost_and_grad(a, circuit_state, angle_params, action_position, b))(
            target_gates[:-1, ::],
            target_gates[1:, ::])
        return jnp.sum(all_err, axis=0), jnp.sum(all_grads, axis=0)

    @partial(jit, static_argnums=(2,))
    def x_opt(circuit_state, angle_params, action_position):
        """
        Calculates the optimal angle parameters.
        :param circuit_state: Collection of integers describing the gate sequence.
        :param angle_params: Collection of angles of each gate.
        :param action_position: Position of the last gate.
        :return: Optimal angle parameters.
        """

        @jit
        def cg(alpha):
            return pvd_cost(circuit_state, alpha, action_position)

        solver = jaxopt.LBFGS(fun=cg, value_and_grad=True, maxiter=max_iter)
        #print(angle_params.shape)
        result = jax.vmap(solver.run)(angle_params)

        return result.state.value, result.params

    return cs_to_unitaries, cost_and_grad, pvd_cost, x_opt


def num_grad(f: Callable):
    """
    Outputs the numerical gradient function.
    :param f: Function.
    :return: Callable, Numerical gradient of f.
    """

    def n_grad(x):

        """
        Calculates the numerical gradient.
        :param x: Function input.
        :return: Numerical gradient of f.
        """

        n_DOF = len(x)
        # print(n_DOF)
        err = f(x)
        dx = np.zeros((n_DOF, *err.shape), np.float64)

        # print("err ", err)
        for k in range(0, n_DOF):
            angles2 = np.zeros(n_DOF)
            for k2 in range(0, n_DOF):
                angles2[k2] = x[k2]
            angles2[k] = x[k] + 1e-9
            dx[k] = -(err - f(angles2)) * 1e9
        return dx

    return n_grad


def main():
    import timeit
    from gate_set_jax import create_fast_ms_set_jax
    from line_profiler import LineProfiler

    np.random.seed(0)
    num_qubits = 8
    gg, gnames, structure = create_fast_ms_set_jax(num_qubits)
    print(gg)
    circuit_state = np.random.randint(1, 3, size=(50,))
    angles_stacked = np.random.randn(30, 5)

    two_angle_gates = ["MS", "Cxy"]
    circuit_state = np.random.randint(1, 4, size=(50,))
    # print(circuit_state)
    angles_init = np.random.randn(50, 2).flatten()
    angles_stacked = np.random.randn(50, 2)
    init_gate = np.identity(2 ** num_qubits, dtype=np.complex128)
    print(angles_stacked.dtype)
    tg = np.identity(2 ** num_qubits, dtype=np.complex128)
    tg = np.array([np.identity(2 ** num_qubits, np.complex128), tg])
    timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
    timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
    cs_to_unitaries, cost_and_grad, pvd_cost, x_opt = pvqd_evo_cost(tg, gg)

    # print(vec_circuit(circuit_state, angles_init, len(circuit_state), init_gate))
    # print(cs_t_us(circuit_state, angles_init, len(circuit_state), init_gate))

    def test():
        return pvd_cost(circuit_state, angles_init, len(circuit_state))

    print(test())
    lp = LineProfiler()
    lp.add_function(cost_and_grad)
    # lp.add_function(cs_t_us)
    lp.add_function(recursive_diff_fidelity)
    lp.add_function(avg_gate_fidelity)
    lp_wrapper = lp(test)
    lp_wrapper()
    lp.print_stats()

    # print(timeit.timeit(test, number=10) / 10)


if __name__ == "__main__":
    main()
