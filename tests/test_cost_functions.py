import jax.numpy as jnp
import numpy as np
from quantum_circuits.gate_set_jax import create_fast_ms_set_jax
from jax.lib import xla_bridge
import jax
from functools import partial

from quantum_circuits.cost_and_grad_jax import create_simple_cost_gates, num_grad
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from quantum_circuits.cost_and_grad_numba import create_cost_gates_standard
from quantum_circuits.layer_env_cost_grad import create_cost_gates_layers
from scipy.linalg import expm


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
    target_gate = gate_set_jax[0]([np.pi / 3, np.pi / 2])[0]
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

    num_qubits = 3
    n_gates = 15
    alpha = np.random.rand(num_qubits * n_gates)
    A = np.random.randn(2 ** num_qubits, 2 ** num_qubits)
    A = A + A.T
    U = expm(1j * A)
    init_gate = np.identity(2 ** num_qubits, np.complex128)
    # print(U.dot(U.conjugate().transpose()))
    gate_set_fast_nb, gate_names_fast_nb, zprod = create_fast_ms_set_numba(num_qubits, z_prod=True)
    circuit_state = np.random.randint(1, 4, n_gates)
    print(circuit_state)
    ctu, cg = create_cost_gates_layers(num_qubits, U, *gate_set_fast_nb)
    cg_circ = cg(circuit_state, alpha, len(circuit_state), init_gate)

    def cost(angles):
        return cg(circuit_state, angles, len(circuit_state), init_gate)[0]

    dx_num = np.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        ffpprec = 1e-9
        shift = np.zeros_like(alpha)
        shift[i] = ffpprec
        shift_angles = alpha + shift
        dx_num[i] = (cost(shift_angles) - cost(alpha)) / ffpprec

    print(dx_num)
    print(cg_circ[1])
    print(np.max(np.abs(cg_circ[1] - dx_num)))
    assert np.max(np.abs(cg_circ[1] - dx_num)) < 1e-6

    gate_z = gate_set_fast_nb[2]
    theta = np.random.rand(num_qubits)
    z_rot, z_rot_grad = gate_z(theta)

    dx_num = np.zeros((theta.shape[0], init_gate.shape[0]), np.complex128)
    ffpprec = 1e-9
    for i in range(theta.shape[0]):
        shift = np.zeros_like(theta)
        shift[i] = ffpprec
        shift_angles = theta + shift
        dx_num[i, ::] = (gate_z(shift_angles)[0] - gate_z(theta)[0]) / ffpprec

    print(z_rot_grad)
    print(dx_num)
    print(np.max(np.abs(z_rot_grad - dx_num)))
    assert np.max(np.abs(z_rot_grad - dx_num)) < 1e-6


if __name__ == '__main__':
    test_cost()
