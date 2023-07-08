import jax
import time
import numpy as np
import string
import jax.numpy as jnp
import matplotlib.pyplot as plt
from quantum_circuits.cost_and_grad_numba import create_cost_gates_standard
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from quantum_circuits.gate_set_jax import create_fast_ms_set_jax
from jax.lib import xla_bridge
from quantum_circuits.cost_and_grad_jax import create_simple_cost_gates
from cycler import cycler

# Runs the comparison of the cost and gradient computation for different number of qubits
# and circuit depths.

def compare_gradients_for_qubits():
    """
    Compare the gradients for different number of qubits
    :return: None
    """

    # Enable gpu and double precision
    # jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_enable_x64', True)

    # os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=15'
    # import jax
    print(jax.devices("gpu"))
    print(xla_bridge.get_backend().platform)

    def compare_gradients(num_qubits, n_gates=50, n_devices=1):

        gg, gnames, _ = create_fast_ms_set_jax(num_qubits)
        gate_set_tuple = gg
        # print(gate_set_tuple)
        target_gate = gate_set_tuple[0]([np.pi / 4, np.pi / 4])[0]
        init_gate = np.identity(2 ** num_qubits, jnp.complex128)
        _, cost_grad, _ = create_simple_cost_gates(target_gate, gate_set_tuple, n_devices=n_devices)
        circuit_state = np.random.randint(1, num_qubits + 2, n_gates)
        angles_init_1 = np.random.randn(len(circuit_state) * 2)
        n_z_gates = (circuit_state >= 2).sum()
        n_other_gates = (circuit_state < 2).sum()
        zero_array = np.zeros(2 * n_z_gates)
        z_angles = np.random.randn(n_z_gates)
        zero_array[1::2] = z_angles
        z_angles = zero_array.reshape(n_z_gates, 2)
        print(np.random.randn(n_other_gates, 2).shape)
        print(z_angles.shape)
        print(angles_init_1.shape)
        # angles_init_1 = np.random.randn(len(circuit_state) * 2)
        # angles_init_1 = np.array([[np.pi / 4, np.pi / 2, ] * n_gates], jnp.float64).flatten()
        angles_init_1 = np.vstack([np.random.randn(n_other_gates, 2), z_angles]).flatten()
        cg_t = cost_grad(circuit_state, angles_init_1, len(circuit_state), init_gate)

        N = 20
        t0 = time.time()
        t = 0
        var_t = 0
        for i in range(N):
            circuit_state = np.random.randint(1, num_qubits + 2, n_gates)
            # angles_init_1 = np.array([[np.pi / 4, np.pi / 2, ] * n_gates], jnp.float64).flatten()
            g1 = cost_grad(circuit_state, angles_init_1, len(circuit_state), target_gate)
            # print(g1[0])
            dt = (time.time() - t0)
            t += dt
            t0 = time.time()
            var_t += dt ** 2

        t_1 = t / N
        std_t1 = np.sqrt(var_t / N - t_1 ** 2) / np.sqrt(N)
        print(t_1, std_t1)
        gg, gnames, _ = create_fast_ms_set_numba(num_qubits)
        cs, cg = create_cost_gates_standard(target_gate, *tuple(gg))
        cg_t = cg(circuit_state, angles_init_1, len(circuit_state), init_gate)

        t0 = time.time()
        t = 0
        var_t = 0
        for i in range(N):
            circuit_state = np.random.randint(1, num_qubits + 2, n_gates)
            # angles_init_1 = np.array([[np.pi / 4, np.pi / 2, ] * n_gates], jnp.float64).flatten()
            g2 = cg(circuit_state, angles_init_1, len(circuit_state), init_gate)
            # print(g2[0])
            dt = (time.time() - t0)
            t += dt
            t0 = time.time()
            var_t += dt ** 2

        t_2 = t / N
        std_t2 = np.sqrt(var_t / N - t_2 ** 2) / np.sqrt(N)

        print(t_2, std_t2)

        return t_1, t_2, std_t1, std_t2

    n_devices_gpu = len(jax.devices())
    n_gates_list_1 = [10, 20, ]
    n_qubits_list_1 = list(range(2, 4))
    t_arr_qubits = np.zeros((4, len(n_qubits_list_1), len(n_gates_list_1), n_devices_gpu))
    n_qubits_list_2 = [4, ]
    n_gates_list_2 = [10, 20, ]
    t_arr_gates = np.zeros((4, len(n_qubits_list_2), len(n_gates_list_2), n_devices_gpu))

    for n_dev in range(n_devices_gpu):
        compare_gradients(num_qubits=3, n_gates=40, n_devices=n_dev+1)

        for i_ng, ng in enumerate(n_gates_list_1):
            for i_nq, nq in enumerate(n_qubits_list_1):
                t_arr_qubits[:, i_nq, i_ng, n_dev] =\
                    np.array(compare_gradients(num_qubits=nq, n_gates=ng, n_devices=n_dev+1))

        compare_gradients(num_qubits=3, n_gates=40, n_devices=n_dev+1)

        for i_ng, ng in enumerate(n_gates_list_2):
            for i_nq, nq in enumerate(n_qubits_list_2):
                t_arr_gates[:, i_nq, i_ng, n_dev] = \
                    np.array(compare_gradients(num_qubits=nq, n_gates=ng, n_devices=n_dev+1))

    # Save gradient data

    np.savetxt("t_arr_qubits.txt", t_arr_qubits.flatten())
    np.savetxt("t_arr_circuit.txt", t_arr_gates.flatten())
    np.savetxt("n_qubits_array_1.txt", np.asarray(n_qubits_list_1))
    np.savetxt("n_gates_array_1.txt", np.asarray(n_gates_list_1))
    np.savetxt("n_gates_array_2.txt", np.asarray(n_gates_list_2))
    np.savetxt("n_qubits_array_2.txt", np.asarray(n_qubits_list_2))


if __name__ == '__main__':
    compare_gradients_for_qubits()
