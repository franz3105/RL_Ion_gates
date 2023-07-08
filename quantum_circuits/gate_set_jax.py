from jax.numpy import array, empty, ones, eye, diag, kron, dot, multiply, divide, append, concatenate
from jax.numpy import flip, transpose, conjugate, real, imag, cos, sin, exp, sqrt, log2, mod, sum, abs, max, min
from jax.numpy import repeat, reshape, cumprod, arange, cumsum, pi, floor, linspace
from jax.numpy import int8, int16, int32, int64, double, complex128
import jax.numpy as jnp
from jax import jit, config
from functools import partial
import numpy as np

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'gpu')


# Necessary Scripts to cache the function efficiently
def all_nchoosek(n):  # Via binomial triangle
    b0 = array([1])
    b = b0[:]
    for i in range(0, n):
        b = concatenate((b0, b[:-1] + b[1:], b0))
    return b


def kronsum_int_1d(h0, n, pow_s):
    h = kron(h0, ones(pow_s[n - 1], int16))
    for i in range(1, n):
        h = h + kron(ones(pow_s[i], int16), kron(h0, ones(pow_s[n - 1 - i], int16)))
    return h


def nkron(h0, n):
    h = h0
    for i in range(n - 1):
        h = kron(h, h0)
    return h


@jit
def Dag(A):
    return conjugate(transpose(A))


def vxv2mat(v):  # Transforms a vector v into a matrix via vector repeats and by summing up repeats in both directions
    M = repeat_reshape(v)
    return M - transpose(M)


def repeat_reshape(v):
    dim = v.shape[0]
    return reshape(repeat(v, dim), (dim, dim))


@jit
def dot_int(A, B):
    return dot(A.astype(double), B.astype(double)).astype(int32)


# dot_int(array([[1,0],[0,1]], int16), array([[0,1],[1,0]], int16))

# Create Cache

def Log_phase_and_Sign_XY(
        n):  # Eigenvector logarithm of phase (vector) and sign (matrix) -> to construct all eigenvectors
    pow_s = tuple(2 ** np.arange(n))
    npe = kronsum_int_1d(array([0, 1], int16), n, pow_s)
    nps = array([[1, 1], [-1, 1]], int16)
    return npe, nkron(nps, n)


def Create_Cache_XY(n):
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
        v_cache = v_cache.at[i, :].set(dot_int(curr_mat, Dag(curr_mat)).flatten())
    e_vals = jnp.sort(jnp.arange(-n, n + 1, 2))
    return phi_cache, v_cache, e_vals


@partial(jit, static_argnums=(3,))
def XY_from_Cache(phi_cache, v_cache, e_vals, n, phi, theta):
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta)
    U = dot(weights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    return reshape(multiply(e_phis[phi_cache], U), (dim, dim))


@partial(jit, static_argnums=(3,))
def XY_from_Cache_and_gradient(phi_cache, v_cache, e_vals, n, phi, theta):
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta / 2)
    dweights = multiply(-1j * e_vals / 2, weights)
    U = dot(weights, v_cache)
    dU_dtheta = dot(dweights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    vC = e_phis[phi_cache]
    arg_ephis = 1j * arange(-n, n + 1)
    dvC_dphi = multiply(vC, arg_ephis[phi_cache])
    dUd_phi = reshape(multiply(dvC_dphi, U), (dim, dim))
    dUd_theta = reshape(multiply(vC, dU_dtheta), (dim, dim))
    U = reshape(multiply(vC, U), (dim, dim))
    return U, dUd_theta, dUd_phi


def Create_Cache_MS(n):
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
        v_cache = v_cache.at[i, :].set(v_cache_xy[i, :] + v_cache_xy[-1 - i, :])
    if eig_num2 < eig_num:
        v_cache = v_cache.at[eig_num - 1, :].set(v_cache_xy[eig_num - 1, :])
    return phi_cache, v_cache, e_vals


@partial(jit, static_argnums=(3,))
def MS_from_Cache(phi_cache, v_cache, e_vals, n, phi, theta):
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta ** 2 / 4)
    U = dot(weights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    return reshape(multiply(e_phis[phi_cache], U), (dim, dim))


def isnan(x):
    if int(x) == -9223372036854775808:
        return True
    else:
        return False


@partial(jit, static_argnums=(3,))
def MS_from_Cache_and_gradient(phi_cache, v_cache, e_vals, n, phi, theta):
    dim = 2 ** n
    weights = exp(-1j * e_vals * theta / 4)
    dweights = multiply(-1j * e_vals / 4, weights)
    U = dot(weights, v_cache)
    dU_dtheta = dot(dweights, v_cache)
    e_phis = exp(1j * arange(-n, n + 1) * phi) / dim
    # print(phi)
    vC = e_phis[phi_cache]
    arg_ephis = 1j * arange(-n, n + 1)
    dvC_dphi = multiply(vC, arg_ephis[phi_cache])
    dUd_phi = reshape(multiply(dvC_dphi, U), (dim, dim))
    dUd_theta = reshape(multiply(vC, dU_dtheta), (dim, dim))
    U = reshape(multiply(vC, U), (dim, dim))
    return U, dUd_theta, dUd_phi


# Necessary Scripts to cache the function efficiently
@partial(jit, static_argnums=(0,))  # Better to rewrite it in lax
def fast_diag_Z_H_ind(i, n):  # Indexes of phases
    s = empty(2 ** n, int8)
    m = 2 ** i
    w = 2 * m
    for j in range(2 ** (n - i)):
        s = s.at[w * j:w * j + m].set(1)
        s = s.at[w * j + m:w * (j + 1)].set(0)  # -1
    return s


def fast_diag_Z_H_inds(i_s, n, l):  # Indexes of phases
    s = np.empty((l, 2 ** n), int16)
    m_s = np.array([2 ** i_s])
    for i in range(l):
        m = m_s[i]
        ns = 2 * m
        for j in range(2 ** (n - i_s)):
            s[i, ns * j:ns * j + m] = 1
            s[i, ns * j + m:ns * (j + 1)] = 0  # -1

    return s


@jit
def Z_dot_U(diag_z, U):
    l = len(diag_z)
    V = empty((l, l), complex128)
    for i in range(l):
        V = V.at[i, :].set(diag_z[i] * U[i, :])
    return V


@jit
def U_dot_Z(U, diag_z):
    l = len(diag_z)
    V = empty((l, l), complex128)
    for i in range(l):
        V = V.at[:, i].set(U[:, i] * diag_z[i])
    return V


def diag_fast_Z_U(i_s, n, theta_s):  # i_s -> indexes of the qubits with z rotation quantum_circuits, n -> number of
    # qubits,
    # theta_s -> angles of rotation for each rot_z
    l = len(i_s)
    v_phi = exp(-1j * theta_s[0] * array([-1, 1]))
    diag_z = v_phi[fast_diag_Z_H_ind(i_s[0], n)]
    for i in range(1, l):
        v_phi = exp(-1j * theta_s[i] * array([-1, 1]))
        diag_z = multiply(diag_z, v_phi[fast_diag_Z_H_ind(i_s[i], n)])
    return diag_z


def diag_fast_Z_U_and_grads(i_s, n, l, theta_s):  # i_s -> indexes of the qubits with z rotation quantum_circuits,
    # n -> number of
    # qubits, theta_s -> angles of rotation for each rot_z
    # print(theta_s)
    grad_mult = array([1j, -1j], jnp.complex128)
    all_inds = fast_diag_Z_H_inds(i_s, n, l)
    v_phi = exp(-1j * theta_s * array([-1, 1], jnp.complex128))
    #print(v_phi)
    diag_z = v_phi[all_inds[0, :]]
    # print(all_inds[0, :])
    grads = grad_mult[all_inds.flatten()].reshape((l, 2 ** n))
    # print(i_s)
    # for i in range(1, l):
    #    v_phi = exp(-1j * theta_s[i] * array([-1, 1]))
    #    diag_z = multiply(diag_z, v_phi[all_inds[i, :]])
    for i in range(l):
        grads = grads.at[i, :].set(multiply(grads[i, :], diag_z))
    new_grad = jnp.diag(grads[0, :])
    #print(diag_z)
    return jnp.diag(diag_z), new_grad, jnp.zeros_like(new_grad)


def fast_multi_Z_and_grads(i_s, n, theta_s):  # i_s -> indexes of the qubits with z rotation quantum_circuits,
    # n -> number of
    # qubits, theta_s -> angles of rotation for each rot_z
    # print(theta_s)
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


def create_fast_ms_set_jax(num_qubits, z_prod=False):
    phi_cache_xy, v_cache_xy, e_vals_xy = Create_Cache_XY(num_qubits)
    phi_cache_ms, v_cache_ms, e_vals_ms = Create_Cache_MS(num_qubits)

    @jit
    def ms_gate_and_grad(theta):
        return MS_from_Cache_and_gradient(phi_cache_ms, v_cache_ms, e_vals_ms, num_qubits, theta[1], theta[0])

    @jit
    def xy_gate_and_grad(theta):
        return XY_from_Cache_and_gradient(phi_cache_xy, v_cache_xy, e_vals_xy, num_qubits, theta[1], theta[0])

    def get_zloc(idx):
        @jit
        def z_iqubit(theta):
            return diag_fast_Z_U_and_grads(idx, num_qubits, 1, theta[0])

        return z_iqubit

    gate_and_grad_set = [ms_gate_and_grad, xy_gate_and_grad] + [get_zloc(k) for k in range(num_qubits)]
    gate_names = ["MS", "Cxy"] + [f"Zloc_{k + 1}" for k in range(num_qubits)]

    return gate_and_grad_set, gate_names, z_prod