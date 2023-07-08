import numpy as np
import qutip as qt
import scipy as sp

# A file containing some target unitary processes.

def randU_Haar(s, rng=np.random.default_rng(0)):
    """
    Generate a random unitary matrix using the Haar measure.
    :param s: Size of the matrix
    :param rng: Random number generator
    :return: Random unitary matrix
    """
    X = rng.normal(1, 1, [s, s]) + 1j * rng.normal(1, 1, [s, s])
    Q, R = np.linalg.qr(X)  # QR decomospition -> Gram-Schmidt orthogonalisation and normalisation
    return Q


def p_matrix(permute_couples):
    """
    Create the permutation matrix for the given permutation couples.
    :param permute_couples: Permutation couples.
    :return: Permutation matrix.
    """

    mat = np.identity(permute_couples.shape[0])
    new_mat = mat.copy()
    for p in permute_couples:
        new_mat[:, p[0]] = mat[:, p[1]]

    return new_mat

def direct_sum(a, b):
    """
    Direct sum of two matrices.
    :param a: Matrix a.
    :param b: Matrix b.
    :return: Direct sum of two matrices.
    """

    dsum = np.zeros(np.add(a.shape, b.shape), np.complex128)
    dsum[:a.shape[0], :a.shape[1]] = a
    dsum[a.shape[0]:, a.shape[1]:] = b
    return dsum

def w_states_matrix(n_qubits):
    """
    Create the matrix for the W states.
    :param n_qubits: Number of qubits.
    :return: Matrix for the W states.
    """
    w_matrix = np.identity(2 ** n_qubits, np.complex128)
    w_gate = np.matrix([[0, 1, 1, 1], [1, 0, -1, 1], [1, 1, 0, -1], [1, -1, 1, 0]], np.complex128)
    S = np.zeros((2**(n_qubits-1), 2**(n_qubits-1)), np.complex128)
    for i in range(2**(n_qubits-1)):
        S[i, 2**(n_qubits-1) - i - 1] = 1
    print(S)
    w_matrix = direct_sum(w_gate, w_gate)
    print(w_matrix)
    #u_trafo = gram_schmidt(w_matrix)
    # print(np.sqrt(n_qubits)*np.round(u_trafo,2))
    return w_matrix/np.sqrt(3)


def gram_schmidt(A):
    """
    Simple Gram-Schmidt orthogonalisation.
    :param A: Matrix of linearly independent column vectors.
    :return: Orthogonalized matrix.
    """
    (n, m) = A.shape

    for i in range(m):

        q = A[:, i]  # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # write the vector back in the matrix
        A[:, i] = q
    # print(A)
    return A


def ucc_operator(n_qubits, alpha):
    """
    Create the unitary operator for the UCC ansatz.
    :param n_qubits: Number of qubits.
    :param alpha: Parameter alpha (aka time of the Hamiltonian).
    :return: Unitary operator.
    """

    sx = qt.sigmax()
    sy = qt.sigmay()
    splus = (sx - 1j * sy) / 2

    Id = qt.identity(2)
    Id_h = qt.tensor([Id] * n_qubits)

    prod = Id_h
    for i in range(n_qubits):
        s_list = [Id] * n_qubits
        s_list[i] = splus
        s_op = qt.tensor(s_list)
        prod = prod * s_op

    prod = prod.full()

    print((prod + prod.conjugate().T))
    return sp.linalg.expm(1j * alpha * (prod + prod.conjugate().T))


def target_unitary(num_qubits, params, t=2 * np.pi):
    """
    Create the target unitary for the XXZ Hamiltonian.
    :param num_qubits: Number of qubits.
    :param t: Time of the Hamiltonian.
    :param params:Parameters.
    :return: Target unitary.
    """
    return xxz(num_qubits, params[0], params[1], params[2], t=t)


def time_dep_cost():
    from quantum_circuits.cost_and_grad_jax import pvqd_evo_cost
    from quantum_circuits.gate_set_jax import create_fast_ms_set_jax
    from jax.lib import xla_bridge
    import jax.numpy as jnp
    import jax
    jax.config.update('jax_enable_x64', True)
    print(jax.devices("gpu"))
    print(xla_bridge.get_backend().platform)
    num_qubits = 3

    gg, gnames = create_fast_ms_set_jax(num_qubits)
    n_gates = 50
    gate_set_tuple = gg
    # print(gate_set_tuple)
    alpha_array = np.linspace(0, 1, 10)
    target_gates = np.array([ucc_operator(num_qubits, a) for a in alpha_array])
    init_gate = np.identity(2 ** num_qubits, jnp.complex128)
    _, cost_grad, pvqd_cost = pvqd_evo_cost(target_gates, gate_set_tuple)
    circuit_state = np.random.randint(1, 3, n_gates)
    # circuit_state = np.array([1,]*n_gates, np.int)
    angles_init_1 = np.array([[np.pi / 3, np.pi / 2, ] * n_gates], jnp.float64).flatten()
    # angles_init_1 = np.array([angles_init_1])
    print(pvqd_cost(circuit_state, angles_init_1, n_gates))


def xxz(num_qubits, h, J, Delta, t=2 * np.pi):
    """
    Create the unitary operator for the XXZ Hamiltonian.
    :param num_qubits:  Number of qubits.
    :param h: Magnetic field.
    :param J: Coupling constant.
    :param Delta: Perturbation.
    :param t: Time.
    :return: Unitary operator.
    """

    d = 2
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    Id = qt.identity(d)
    Id_h = qt.tensor([Id] * num_qubits)

    h0 = qt.tensor([qt.qzero(d)] * num_qubits)
    for i_s, s in enumerate([sz]):
        for i in range(num_qubits):
            Id_list = [Id] * num_qubits
            op_i = Id_list.copy()
            op_i[i] = s
            h0 += -2 * h * qt.tensor(op_i)

    for i_s, s in enumerate([sx, sy]):
        for i in range(0, num_qubits - 1):
            j = i + 1
            print(i, j)
            Id_list = [Id] * num_qubits
            op_ij = Id_list.copy()
            op_ij[i] = s
            op_ij[j] = s
            h0 += -J * qt.tensor(op_ij)

    for i_s, s in enumerate([sz]):
        for i in range(0, num_qubits - 1):
            j = i + 1
            Id_list = [Id] * num_qubits
            op_ij = Id_list.copy()
            op_ij[i] = s
            op_ij[j] = s
            # print(dZZ)
            h0 += -J * Delta * (qt.tensor(op_ij) - 1 / 4 * Id_h)

    return sp.linalg.expm(-h0.full() * 1j * t)


def threequbit_model(oqc_params):
    """
    Create the three qubit model.
    :param oqc_params:  Parameters.
    :return: Model.
    """
    num_qubits = 3
    d = 2
    Hdim = d ** num_qubits
    oqc_params = np.array(oqc_params)
    Delta1 = 2 * np.pi * 0.3
    Delta2 = 2 * np.pi * 0.2
    alpha_1 = 2 * np.pi * (-0.35)
    alpha_2 = 2 * np.pi * (-0.34)

    J1 = np.pi * 0.01
    J2 = np.pi * 0.01
    K = 1
    I = qt.qeye(2)
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    # print(10 * qt.tensor([I, I, Z]))

    H_0 = qt.tensor([qt.qzero(d)] * num_qubits)
    H_0 += Delta1 * qt.tensor([Z, I, I]) + Delta2 * qt.tensor([I, Z, I])

    H_J = J1 * (qt.tensor([X, X, I]) + qt.tensor([Y, Y, I]) + qt.tensor([I, X, X])) + \
          J2 * (qt.tensor([I, X, X]) + qt.tensor([I, Y, Y]) + qt.tensor([I, Y, Y])) + \
          qt.tensor([X, X, X]) + qt.tensor([Z, Z, Z])

    # print(H_J)
    H_0 += H_J
    Hc_list = []
    Hc_names_list = []
    control_gates = dict(X=K * qt.sigmax(), Y=K * qt.sigmay())
    h_params = np.array([Delta1, Delta2, J1, J2])

    for i in [0, 1, 2]:
        for sigma in control_gates:
            qudit_list = [qt.qeye(d)] * num_qubits
            qudit_list[i] = control_gates[sigma]
            Hc_list.append(qt.tensor(qudit_list))
            generic_name = ["I", ] * num_qubits
            generic_name[i] = sigma
            Hc_names_list.append("".join(generic_name))

    return H_0.full()


def main():
    params = (0.0, np.pi, 1)
    nq = 4
    u = target_unitary(nq, params, t=1)
    print(np.round(u, 2))
    # print(sp.linalg.expm(h0*1j*1))
    print(np.abs(np.trace(u.dot(np.identity(2 ** nq)))) ** 2 / (2 ** (2 * nq)))

    return


if __name__ == "__main__":
    main()
