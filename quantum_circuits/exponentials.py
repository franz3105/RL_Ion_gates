# -*- coding: utf-8 -*-

from numba import njit
from numpy import *
from numpy.linalg import eigh


@njit(fastmath=True, nogil=True)
def Dag(U):

    """
    Computes the adjoint of a unitary matrix U.
    :param U: unitary matrix.
    :return: adjoint of U.
    """

    return transpose(conj(U))


@njit(fastmath=True, nogil=True)
def Commutator(A, B):

    """
    Computes the commutator of two matrices A and B.
    :param A: matrix.
    :param B: matrix.
    :return: commutator of A and B.
    """

    return dot(A, B) - dot(B, A)


@njit(fastmath=True, nogil=True)
def vecxvec(E, wf, alpha):  # Many times faster than multiply
    """
    Computes the exponential .
    :param E:   eigenvalue vector.
    :param wf:  collections of eigenvectors.
    :param alpha:   Parameter.
    :return:
    """

    s = E.size

    for i in range(s):
        wf[i] = exp(-1j * E[i] * alpha) * wf[i]

    return wf


# Construct matrix exponentials and products faster than in numpy or scipy

@njit(fastmath=True, nogil=True)
def vm_exp_mul(E, V, dt=1.0):  # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    """

    :param E:   eigenvalue vector.
    :param V:   eigenvector matrix.
    :param dt:  time step.
    :return:    expm(diag(vector))*matrix
    """

    s = E.size

    A = empty((s, s), complex128)

    for i in range(s):
        A[i, :] = exp(-1j * E[i] * dt) * Dag(V[:, i])

    return A


@njit(fastmath=True, nogil=True)
def expmH_from_Eig(E, V, dt=1.0):

    """

    :param E:   eigenvalue vector.
    :param V:   eigenvector matrix.
    :param dt:  time step.
    :return:    expm(-1j*H*dt)
    """

    U = dot(V, vm_exp_mul(E, V, dt))

    return U


@njit(fastmath=True, nogil=True)
def expmH(H, dt=1.0):

    """

    :param H:   Hamiltonian.
    :param dt:  time step.
    :return:    expm(-1j*H*dt)
    """

    E, V = eigh(H)

    return expmH_from_Eig(E, V, dt)
