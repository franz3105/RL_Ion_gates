import numpy as np

from quantum_circuits.gate_set_jax import create_fast_ms_set_jax
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from quantum_circuits.quantum_gates_numba import create_standard_ms_set
from quantum_circuits.gate_set_numba import Create_Cache_XY

# Tests the gates both in numba and jax. In particular it compares the standard exponentiation of the Hamiltonians
# with the spectral decomposition of the Hamiltonians. The latter is used in the fast gates.

def test_gates():

    n_list = range(2, 6)

    for n in n_list:
        d = 2**n
        B = np.random.normal(size=(d ** 2,)).reshape(d, d)
        C = np.arange(-n, n + 1)[Create_Cache_XY(n)[0]].reshape(d, d)
        print(C)

        # v = Create_Cache_XY(n)[1]
        # diag_evals = Create_Cache_XY(n)[2]
        # print(np.dot(diag_evals, v))
        # print(C)
        # print(C.dot(C).dot(C) + 256*C)

        gate_set_jax, gate_names_jax, zprod = create_fast_ms_set_jax(n)
        gate_set_fast_nb, gate_names_fast_nb, zprod = create_fast_ms_set_numba(n)
        gate_set_nb, gate_names_nb, gate_grad_nb = create_standard_ms_set(n)

        ms_jax = gate_set_jax[0]
        ms_fast_nb = gate_set_fast_nb[0]
        ms_nb = gate_set_nb[0]

        xy_jax = gate_set_jax[1]
        xy_fast_nb = gate_set_fast_nb[1]
        xy_nb = gate_set_nb[1]

        U = np.identity(2 ** n)
        print('n=' + str(n))

        for i in range(0, n):
            z_jax = gate_set_jax[2 + i]
            z_fast_nb = gate_set_fast_nb[2]
            z_nb = gate_set_nb[2:][::-1][i]

            theta = 2 * np.pi * np.random.randn()

            z_1 = z_jax([theta])[0]

            z_2 = np.diag(z_fast_nb(theta, i)[0])
            #print(z_1)
            z_3 = z_nb(theta)
            #print(z_3)
            #print(z_1 - z_3)
            Z = np.array([[1, 0], [0, -1]], np.complex128)
            I = np.identity(2, np.complex128)
            #print(expm(-1j * theta * np.kron(Z, I)))
            #print(np.kron(I,Z))

            print(f"Z_{i} fast jax vs Z_{i} fast numba: {np.amax(z_1 - z_2)}")
            assert np.amax(z_1 - z_2) < 1e-10
            print(f"Z_{i} fast jax vs Z_{i} numba: {np.amax(z_1 - z_3)}")
            assert np.amax(z_1 - z_3) < 1e-10

        theta = 2 * np.pi * np.random.randn()
        phi = 2 * np.pi * np.random.randn()

        ms_1, *_ = ms_jax(np.array([theta, phi]))
        ms_2, *_ = ms_fast_nb(theta, phi)
        ms_3 = ms_nb(theta, phi)

        print(f"MS fast jax vs MS fast numba: {np.amax(ms_1 - ms_2)}")
        assert np.amax(ms_1 - ms_2) < 1e-10
        print(f"MS fast jax vs MS numba: {np.amax(ms_1 - ms_3)}")
        assert np.amax(ms_1 - ms_3) < 1e-10

        _, _, msg_1 = ms_jax(np.array([theta, phi]))
        _, msg_2, _ = ms_fast_nb(theta, phi)

        print(f"MS fast jax gradient vs MS fast numba gradient: {np.amax(msg_1 - msg_2)}")
        assert np.amax(msg_1 - msg_2) < 1e-10
        xy_1, *_ = xy_jax(np.array([theta, phi]))
        xy_2, *_ = xy_fast_nb(theta, phi)
        xy_3 = xy_nb(theta, phi)

        print(f"XY fast jax vs XY fast numba: {np.amax(xy_1 - xy_2)}")
        assert np.amax(xy_1 - xy_2) < 1e-10
        print(f"XY fast jax vs XY numba: {np.amax(xy_1 - xy_3)}")
        assert np.amax(xy_1 - xy_3) < 1e-10

        # a = np.diag(diag_fast_Z_U((i,), n, theta))
        # print(Zloc(theta, i, n))
        # print(a)
        # print(np.max(Zloc(theta, i, n) - a))
        #    print(np.max(Z_dot_U(a, U) - Zloc(theta, i, n).dot(U)))
        #    print(np.max(U_dot_Z(U, a) - U.dot(Zloc(theta, i, n))))
