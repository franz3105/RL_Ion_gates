import numpy as np
import sympy as sy
from numpy import array, empty, ones, kron, dot, multiply, concatenate, transpose, conjugate, exp, mod, repeat, \
    reshape, arange, cumsum, int8, int16, int32, double, complex128
from numba import njit
from typing import List, Union, Tuple, Callable
from numpy.linalg import eigh


@njit(fastmath=True, nogil=True)
def tensor(op_list: List) -> np.ndarray:
    """
    :param op_list: list of operators.
    :return: tensor product of operators.
    """
    s = op_list[0]

    for i_op in range(1, len(op_list)):
        s = np.kron(s, op_list[i_op])

    return s


@njit(fastmath=True, nogil=True)
def Dag(U: np.ndarray) -> np.ndarray:
    """
    :param U: unitary matrix.
    :return: dagger of U.
    """
    U_dag = np.conjugate(U).T
    return U_dag


# Construct matrix exponentials and products faster than in numpy or scipy
@njit(fastmath=True, nogil=True)
def vm_exp_mul(E: np.ndarray, V: np.ndarray, dt=1.0) -> np.ndarray:  # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    """
    Computes matrix first part of the matrix exponential and products faster than in numpy or scipy.
    :param E: Eigevalues.
    :param V: Eigenvector matrix.
    :param dt: Time step.
    :return: exp(E)V.
    """
    s = E.size
    A = np.empty((s, s), np.complex128)
    for k in range(s):
        C = -1j * E[k] * dt
        A[k, :] = np.exp(C) * Dag(V[:, k])

    return A


@njit(fastmath=True)
def expmH_from_Eig(E: np.ndarray, V: np.ndarray, dt=1.0) -> np.ndarray:
    """
    Computes the matrix exponential and products faster than in numpy or scipy.
    :param E: Eigenvalues.
    :param V: Eigenvector matrix.
    :param dt: Time step.
    :return: Exponential matrix computed via spectral decomposition V exp(E)V.T.conj()
    """
    U = np.dot(V, vm_exp_mul(E, V, dt))
    return U


@njit(fastmath=True)
def get_eigh(H: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues and eigenvectors of a Hermitian matrix.
    :param H: Hermitian matrix.
    :return: eigenvalues and eigenvectors.
    """
    return np.linalg.eigh(1j * H)


@njit(fastmath=True)
def expmH(H: np.ndarray, dt=1) -> np.ndarray:
    """
    Computes the matrix exponential and products faster than in numpy or scipy.
    :param H: Hermitian matrix.
    :param dt: Time step.
    :return: Exponential matrix computed via spectral decomposition V exp(E)V.T.conj()
    """
    E, V = get_eigh(H)
    U = expmH_from_Eig(E, V, dt)
    return U


# Necessary Scripts to cache the function efficiently
@njit(fastmath=True, nogil=True)
def all_nchoosek(n: int) -> np.ndarray:  # Via binomial triangle
    """
    Computes all possible combinations of n elements.
    :param n: Number of elements.
    :return: All possible combinations of n elements.
    """
    b0 = array([1])
    b = b0[:]
    for i in range(0, n):
        b = concatenate((b0, b[:-1] + b[1:], b0))
    return b


@njit(fastmath=True, nogil=True)
def kronsum_int_1d(h0: np.ndarray, n: int) -> np.ndarray:
    """
    Computes the kronecker product of h0 and all possible combinations of n elements.
    :param h0: Initial vector.
    :param n: Number of elements.
    :return: kronecker product of h0 and all possible combinations of n elements.
    """
    s = h0.shape[0]
    pow_s = s ** arange(n)
    h = kron(h0, ones(pow_s[n - 1], int16))
    for i in range(1, n):
        h = h + kron(ones(pow_s[i], int16), kron(h0, ones(pow_s[n - 1 - i], int16)))
    return h


@njit(fastmath=True, nogil=True)
def nkron(h0: np.ndarray, n: int) -> np.ndarray:
    """
     Returns the kronecker product of h0 and n times.
    :param h0: matrix (np.ndarray).
    :param n: number of times the matrix is tensorized.
    :return: kronecker product of h0 with itself n times.
    """
    h = h0.copy()
    for i in range(n - 1):
        h = kron(h, h0)
    return h


@njit(fastmath=True, nogil=True)
def Dag(A: np.ndarray) -> np.ndarray:
    return conjugate(transpose(A))


@njit(fastmath=True, nogil=True)
def vxv2mat(v: np.ndarray) -> np.ndarray:
    """
    Transforms a vector v into a matrix via vector repeats and by summing up repeats in both directions.
    :param v: vector (np.ndarray).
    :return: matrix (np.ndarray).
    """
    M = repeat_reshape(v)
    return M - transpose(M)


@njit(fastmath=True, nogil=True)
def repeat_reshape(v: np.ndarray) -> np.ndarray:
    dim = v.shape[0]
    return reshape(repeat(v, dim), (dim, dim))


@njit(fastmath=True, nogil=True)
def dot_int(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return dot(A.astype(double), B.astype(double)).astype(int32)


# dot_int(array([[1,0],[0,1]], int16), array([[0,1],[1,0]], int16))

# Create Cache
@njit(fastmath=True, nogil=True)
def Log_phase_and_Sign_XY(
        n: int):
    """
    Eigenvector logarithm of phase (vector) and sign (matrix) -> to construct all eigenvectors.
    :param n: size of vector.
    :return: logarithm of phase and sign of eigenvectors.
    """
    npe = kronsum_int_1d(array([0, 1], int16), n)
    nps = array([[1, 1], [-1, 1]], int16)
    return npe, nkron(nps, n)


@njit(fastmath=True, nogil=True)
def Create_Cache_XY(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the cached matrices needed for the XY gate.
    :param n: Number of qubits.
    :return: Phase vector, eigenvectors and eigenvalues.
    """
    # Determine log phase vector and sign matrix of the eigenvectors
    dim = 2 ** n
    log_ephi, signs = Log_phase_and_Sign_XY(n)
    signs = signs
    # Sort the indexes of eigenvectors with common eigenvalue (degenerate groups)
    eig_val_indexes = np.argsort(log_ephi)
    n_d = all_nchoosek(n)  # degeneracy of eigenvalues
    subspace_boundary_indexes = concatenate((array([0]), cumsum(n_d)))
    phi_cache = (vxv2mat(log_ephi) + n).flatten()
    v_cache = empty((n + 1, dim ** 2), complex128)
    for i in range(n + 1):
        curr_inds = eig_val_indexes[subspace_boundary_indexes[i]:subspace_boundary_indexes[i + 1]]
        curr_mat = signs[:, curr_inds]
        v_cache[i, :] = dot_int(curr_mat, Dag(curr_mat)).flatten()
    e_vals = np.sort(np.arange(-n, n + 1, 2))
    return phi_cache, v_cache, e_vals


@njit(fastmath=True, nogil=True)
def XY_from_Cache(phi_cache: np.ndarray, v_cache: np.ndarray, e_vals: np.ndarray,
                  n: int, phi: np.float64, theta: np.float64) -> np.ndarray:
    """
    Computes the XY gate expansion of the eigenvectors in the eigenbasis.
    :param phi_cache: Cached phase vector.
    :param v_cache: Cached eigenvectors.
    :param e_vals: Cached eigenvalues.
    :param n: Number of qubits.
    :param phi: Phase angle of the MS gate.
    :param theta: Rotation angle of the MS gate.
    :return: XY gate and its gradients with respect to phi and theta, obtained with spectral decomposition.
    """
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta)
    U = dot(weights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    return reshape(multiply(e_phis[phi_cache], U), (dim, dim))


@njit(fastmath=True, nogil=True)
def XY_from_Cache_and_gradient(phi_cache: np.ndarray, v_cache: np.ndarray, e_vals: np.ndarray, n: int, phi: np.float64
                               , theta: np.float64) -> Tuple[np.ndarray]:
    """
    Computes the spectral decomposition of the XY gate.
    :param phi_cache: Cached phase vector.
    :param v_cache: Cached eigenvectors.
    :param e_vals: Cached eigenvalues.
    :param n: Number of qubits.
    :param phi: Phase angle of the MS gate.
    :param theta: Rotation angle of the MS gate.
    :return: XY gate and its gradients with respect to phi and theta, obtained with spectral decomposition.
    """
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta / 2)
    dweights = multiply(-1j * e_vals / 2, weights)
    U = dot(weights, v_cache)
    dU_dtheta = dot(dweights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    vC = np.take(e_phis, phi_cache)
    arg_ephis = 1j * arange(-n, n + 1)
    dvC_dphi = multiply(vC, np.take(arg_ephis, phi_cache))
    dUd_phi = reshape(multiply(dvC_dphi, U), (dim, dim))
    dUd_theta = reshape(multiply(vC, dU_dtheta), (dim, dim))
    U = reshape(multiply(vC, U), (dim, dim))
    return U, dUd_phi, dUd_theta


@njit(fastmath=True, nogil=True)
def Create_Cache_MS(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the cached matrices needed for the MS gate.
    :param n: Number of qubits.
    :return: Phase vector, eigenvectors and eigenvalues.
    """
    dim = 2 ** n
    phi_cache, v_cache_xy, e_vals_xy = Create_Cache_XY(n)
    e_vals = arange(-n, 1, 2) ** 2
    eig_num = e_vals.shape[0]
    v_cache = empty((eig_num, dim ** 2), complex128)
    if mod(n, 2):
        eig_num2 = e_vals.shape[0]
    else:
        eig_num2 = eig_num - 1
    for i in range(eig_num2):
        v_cache[i, :] = v_cache_xy[i, :] + v_cache_xy[-1 - i, :]
    if eig_num2 < eig_num:
        v_cache[eig_num - 1, :] = v_cache_xy[eig_num - 1, :]
    return phi_cache, v_cache, e_vals


@njit(fastmath=True, nogil=True)
def MS_from_Cache(phi_cache: np.ndarray, v_cache: np.ndarray, e_vals: np.ndarray, n: int,
                  phi: np.float64, theta: np.float64) -> np.ndarray:
    """
    Computes the spectral decomposition of the MS gate.
    :param phi_cache: Cached phase vector.
    :param v_cache: Cached eigenvectors.
    :param e_vals: Cached eigenvalues.
    :param n: Number of eigenvalues.
    :param phi: Phase angle of the MS gate.
    :param theta: Rotation angle of the MS gate.
    :return: MS gate obtained with spectral decomposition.
    """
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta ** 2 / 4)
    U = dot(weights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    return reshape(multiply(e_phis[phi_cache], U), (dim, dim))


@njit(fastmath=True, nogil=True)
def MS_from_Cache_and_gradient(phi_cache: np.ndarray, v_cache: np.ndarray, e_vals: np.ndarray, n: int,
                               phi: np.float64, theta: np.float64) -> Tuple[np.ndarray]:
    """
    Computes the spectral decomposition of the MS gate.
    :param phi_cache: Cached phase vector.
    :param v_cache: Cached eigenvectors.
    :param e_vals: Cached eigenvalues.
    :param n: Number of qubits.
    :param phi: Phase angle of the MS gate.
    :param theta: Rotation angle of the MS gate.
    :return: MS gate and its gradients with respect to phi and theta, obtained with spectral decomposition.
    """
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta / 4)
    dweights = multiply(-1j * e_vals / 4, weights)
    U = dot(weights, v_cache)
    dU_dtheta = dot(dweights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    # print(phi)
    vC = np.take(e_phis, phi_cache)
    arg_ephis = 1j * arange(-n, n + 1)
    dvC_dphi = multiply(vC, np.take(arg_ephis, phi_cache))
    dUd_phi = reshape(multiply(dvC_dphi, U), (dim, dim))
    dUd_theta = reshape(multiply(vC, dU_dtheta), (dim, dim))
    U = reshape(multiply(vC, U), (dim, dim))
    return U, dUd_phi, dUd_theta


# Necessary Scripts to cache the function efficiently
@njit(fastmath=True, nogil=True)
def fast_diag_Z_H_ind(i: int, n: int) -> np.ndarray:  # Indexes of phases
    """
    Fast Z gate with indexing.
    :param i: Target qubit.
    :param n: Total number of qubits.
    :return: Matrix of the indices where the phase shift is applied.
    """
    s = empty(2 ** n, int8)
    m = 2 ** i
    w = 2 * m
    for j in range(2 ** (n - i)):
        s[w * j:w * j + m] = 1
        s[w * j + m:w * (j + 1)] = 0  # -1
    return s


@njit(fastmath=True, nogil=True)
def fast_diag_Z_H_inds(i_s: np.ndarray, n: int) -> np.ndarray:  # Indexes of phases
    """
    Computes the single qubit Z gate acting on multiple qubits defined by i_s.
    :param i_s: Indexes of qubits.
    :param n: Number of qubits.
    :return: Matrix of the indices where the phase shift is applied.
    """
    L = len(i_s)
    s = empty((L, 2 ** n), int16)
    m_s = 2 ** i_s
    for i in range(L):
        m = m_s[i]
        ns = 2 * m
        for j in range(2 ** (n - i_s[i])):
            s[i, ns * j:ns * j + m] = 1
            s[i, ns * j + m:ns * (j + 1)] = 0  # -1

    return s


@njit(fastmath=True, nogil=True)
def z_dot_v(diag_z, v):
    """

    :param diag_z:
    :param v:
    :return:
    """
    return diag_z * v


@njit(fastmath=True, nogil=True)
def Z_dot_U(diag_z, U):
    V = U * np.expand_dims(diag_z, axis=1)
    return V


@njit(fastmath=True, nogil=True)
def U_dot_Z(U, diag_z):
    V = np.expand_dims(diag_z, axis=0) * U
    return V


@njit(fastmath=True, nogil=True)
def diag_fast_Z_U(i_s, n, theta_s):
    """
    Computes the single qubit Z gate acting on multiple qubits defined by i_s
    :param i_s: Indexes of the qubits with z rotation quantum_circuits.
    :param n: Number of qubits.
    :param theta_s: Angles of rotation for each rot_z.
    :return: Matrix of the indices where the phase shift is applied.
    """

    l = len(i_s)
    v_phi = exp(-1j * theta_s[0] * array([-1, 1]))
    # print(fast_diag_Z_H_ind(i_s[0], n))
    diag_z = np.take(v_phi, fast_diag_Z_H_ind(i_s[0], n))
    for i in range(1, l):
        v_phi = exp(-1j * theta_s[i] * array([-1, 1]))
        # print(i_s[i], n)
        diag_z = multiply(diag_z, np.take(v_phi, fast_diag_Z_H_ind(i_s[i], n)))
    return diag_z


@njit(fastmath=True, nogil=True)
def fast_single_Z_and_grads(i_s: np.ndarray, n: int, theta_s: np.ndarray):
    """
    Computes the single qubit Z gate acting on one single qubit defined by i_s and its gradient
    (Only one gradient is considered here).
    :param i_s: Index of the qubit with z rotation on the quantum_circuits.
    :param n: Number of qubits.
    :param theta_s: Angles of rotation for each rot_z.
    :return: Matrix of the indices where the phase shift is applied.
    """
    l = len(i_s)
    grad_mult = array([1j, -1j], np.complex128)
    all_inds = fast_diag_Z_H_inds(i_s, n)
    v_phi = exp(-1j * theta_s[0] * array([-1, 1], np.complex128))
    # print(v_phi)
    diag_z = np.take(v_phi, all_inds[0, :])
    # print(all_inds[0, :])
    grads = np.take(grad_mult, all_inds.flatten()).reshape((l, 2 ** n))
    # print(i_s)
    # for i in range(1, l):
    #    v_phi = exp(-1j * theta_s[i] * array([-1, 1]))
    #    diag_z = multiply(diag_z, v_phi[all_inds[i, :]])
    for i in range(l):
        grads[i, :] = multiply(grads[i, :], diag_z)

    return diag_z, grads[0, :]


@njit(fastmath=True, nogil=True)
def fast_multi_Z_and_grads(i_s: np.ndarray, n: int, theta_s: np.ndarray):
    """
    Computes the single qubit Z gate acting on multiple qubits defined by i_s and its gradients.
    :param i_s: Indexes of the qubits with z rotation on the quantum_circuits.
    :param n: Number of qubits.
    :param theta_s: Angles of rotation for each rot_z.
    :return: Matrix of the indices where the phase shift is applied.
    """
    l = len(i_s)
    grad_mult = array([1j, -1j], np.complex128)
    all_inds = fast_diag_Z_H_inds(i_s, n)
    v_phi = exp(-1j * theta_s[0] * array([-1, 1], np.complex128))
    diag_z = v_phi[all_inds[0, :]]
    grads = grad_mult[all_inds.flatten()].reshape((l, 2 ** n))

    for i in range(1, l):
        v_phi = exp(-1j * theta_s[i] * array([-1, 1]))
        diag_z = multiply(diag_z, v_phi[all_inds[i, :]])

    for j in range(l):
        grads[j, :] = multiply(grads[j, :], diag_z)

    return diag_z, grads


def z_cache(n):
    # Not very useful tbh. No idea why it is here.
    return np.eye(2 ** n)


@njit(fastmath=True, nogil=True)
def Z_and_grads(idx: int, theta: np.float64, n: int) -> np.ndarray:
    """
    Z gate computed with spectral decomposition.
    :param idx: Target qubit.
    :param theta:  Angles of rotation.
    :param n:   Number of qubits.
    :return: Unitary of the Z gate.
    """
    idx_arr = fast_diag_Z_H_ind(idx, n)
    z_factor = expmH(-1j * theta * array([-1, 1]))
    z_u = z_factor[idx_arr]

    return z_u


def create_fast_ms_set_numba(num_qubits, z_prod=False) -> \
        Tuple[List[Callable], List[str], bool]:

    """
    Generating function for the fast gate set.
    :param num_qubits:  Number of qubits.
    :param z_prod: What type of Z gate to use (Collective=True, Single-qubits=False).
    :return: List of the gate functions.
    """
    phi_cache_xy, v_cache_xy, e_vals_xy = Create_Cache_XY(num_qubits)
    phi_cache_ms, v_cache_ms, e_vals_ms = Create_Cache_MS(num_qubits)
    all_indices = np.arange(0, num_qubits)

    @njit(fastmath=True, nogil=True)
    def ms_gate_and_grad(theta, phi):
        return MS_from_Cache_and_gradient(phi_cache_ms, v_cache_ms, e_vals_ms, num_qubits, phi, theta)

    @njit(fastmath=True, nogil=True)
    def xy_gate_and_grad(theta, phi):
        return XY_from_Cache_and_gradient(phi_cache_xy, v_cache_xy, e_vals_xy, num_qubits, phi, theta)

    if not z_prod:
        def get_zloc():

            @njit(fastmath=True, nogil=True)
            def z_iqubit(theta, idx):
                idx_arr = np.array([idx], np.int8)

                # print(num_gates)
                # print(theta.dtype)
                theta = np.array([theta], np.float64)
                return fast_single_Z_and_grads(idx_arr, num_qubits, theta)

            return z_iqubit

        gate_and_grad_set = [ms_gate_and_grad, xy_gate_and_grad] + [get_zloc()]
        gate_names = ["MS", "Cxy"] + [f"Zloc_{k + 1}" for k in range(num_qubits)]
    else:
        def get_zloc():

            @njit(fastmath=True, nogil=True)
            def z_iqubit(theta):
                return fast_multi_Z_and_grads(all_indices, num_qubits, theta)

            return z_iqubit

        gate_and_grad_set = [ms_gate_and_grad, xy_gate_and_grad, get_zloc()]
        gate_names = ["MS", "Cxy", "Z"]

    return gate_and_grad_set, gate_names, z_prod


def num_grad(f: Callable) -> Callable:
    """
    Another numerical gradient.
    :param f: function.
    :return: Function calculating the numerical gradient.
    """
    def n_grad(x):
        n_DOF = len(x)
        # print(n_DOF)
        err = f(x)
        dx = np.zeros((n_DOF, *err.shape), np.complex128)

        # print("err ", err)
        for k in range(0, n_DOF):
            angles2 = np.zeros(n_DOF)
            for k2 in range(0, n_DOF):
                angles2[k2] = x[k2]
            angles2[k] = x[k] + 1e-9
            dx[k, ::] = -(err - f(angles2)) * 1e9
        return dx

    return n_grad