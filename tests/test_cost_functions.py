import jax.numpy as jnp
import numpy as np
from quantum_circuits.gate_set_jax import create_fast_ms_set_jax
from jax.lib import xla_bridge
import jax
from functools import partial

from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from quantum_circuits.cost_and_grad_numba import create_cost_gates_standard
from quantum_circuits.cost_and_grad_jax import create_simple_cost_gates, num_grad


def test_cost():

    jax.config.update('jax_enable_x64', True)
    jax.disable_jit()
    # os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=15'
    # import jax
    # print(jax.devices("gpu"))
    # print(xla_bridge.get_backend().platform)
    num_qubits = 4

    gate_set_jax, g_names_jax, _ = create_fast_ms_set_jax(num_qubits)
    gate_set_nb, g_names_nb, _ = create_fast_ms_set_numba(num_qubits, z_prod=False)
    f = gate_set_nb[2]
    gate_set_nb_2 = gate_set_nb[:2] + [partial(f, idx=i_q) for i_q in range(num_qubits)]

    n_gates = 10
    # print(gate_set_tuple)
    theta = np.random.randn()
    phi = np.random.randn()
    target_gate = gate_set_jax[0]([2 * np.pi, np.pi / 2])[0]
    init_gate = np.identity(2 ** num_qubits, jnp.complex128)
    cs_jax, cost_grad_jax, *_ = create_simple_cost_gates(target_gate, gate_set_jax)
    print(*gate_set_nb)
    cs_numba, cost_grad_numba, *_ = create_cost_gates_standard(target_gate, *gate_set_nb)
    circuit_state = np.random.randint(1, num_qubits + 2, n_gates)
    # print(circuit_state)
    # circuit_state = np.array([2,]*n_gates)
    angle_count = 0
    al = []
    for i_g in range(n_gates):
        idx = circuit_state[i_g] - 1
        if idx < 2:
            al.append(theta)
            al.append(phi)
            angle_count += 2
        else:
            al.append(theta)
            angle_count += 1

    # angles_init_1 = np.random.randn(angle_count)

    angles_init_1 = np.array([[theta, phi, ], ] * n_gates, jnp.float64)
    l1 = []
    l2 = []
    # print(angles_init_1)
    # print(angles_init_1[:, 0])
    # print(angles_init_1[:, 1])
    angles_init_2 = np.append(angles_init_1[:, 0], angles_init_1[:, 1]).flatten()
    # print(angles_init_1.reshape(len(circuit_state), 2, order="C"))
    # print(angles_init_2)
    # angles_init_1 = angles_init_1.flatten()
    angles_init_1 = np.array(al).flatten()

    U = init_gate
    # print(init_gate)
    angle_count = 0
    for i_g in range(n_gates):
        idx = circuit_state[i_g] - 1
        # print(theta, phi)
        if idx < 2:
            theta, phi = angles_init_1[angle_count], angles_init_1[angle_count + 1]
            gate = gate_set_nb_2[idx](theta, phi)[0]
            angle_count += 2

        else:
            theta = angles_init_1[angle_count]
            gate = np.diag(gate_set_nb_2[idx](theta)[0])
            angle_count += 1

            # print(idx, theta)
            # print(gate)
        # print(gate)
        U = np.dot(U, gate)

    # print(U)

    def cost_jax(angles):
        return cost_grad_jax(circuit_state, angles, len(circuit_state), init_gate)[0]

    def cost_numba(angles):
        return cost_grad_numba(circuit_state, angles, len(circuit_state), init_gate)[0]

    U_jax = cs_jax(circuit_state, angles_init_2, init_gate)[0]
    U_numba = cs_numba(circuit_state, angles_init_1, len(circuit_state), init_gate)[3]
    print(np.amax(U_numba - U))
    print(f"U_numba vs U_jax: {np.amax(U_numba - U_jax)}")

    print(cost_jax(angles_init_2))
    print(cost_numba(angles_init_1))
    print(np.amax(cost_jax(angles_init_2) - cost_numba(angles_init_1)))
    assert np.allclose(cost_jax(angles_init_2), cost_numba(angles_init_1))
    jax_num_grad = num_grad(cost_jax)
    numba_num_grad = num_grad(cost_numba)

    # print(new_cost_numgrad(angles_init_1))
    # print(cost_grad(circuit_state, angles_init_1, len(circuit_state), init_gate)[1])
    print(angles_init_1.shape)
    print(angles_init_2.shape)
    cg_jax = cost_grad_jax(circuit_state, angles_init_2, len(circuit_state), init_gate)[1]
    cg_numba = cost_grad_numba(circuit_state, angles_init_1, len(circuit_state), init_gate)[1]
    print(cg_numba)
    print(cg_jax)
    # print(cg_jax.T.ravel())
    # print(np.append(cg_jax[:, 0], cg_jax[:, 1]))
    # print(jax_num_grad(angles_init_2.flatten()))
    # print(np.amax(cg_numba - cg_jax))
    print(np.amax(np.abs(jax_num_grad(angles_init_2.flatten()) - cg_jax.flatten())))
    assert np.amax(np.abs(jax_num_grad(angles_init_2.flatten()) - cg_jax.flatten())) < 1e-6
    print(np.amax(np.abs(numba_num_grad(angles_init_1) - cg_numba)))
    assert np.amax(np.abs(numba_num_grad(angles_init_1) - cg_numba)) < 1e-6



if __name__ == '__main__':
    test_cost()
