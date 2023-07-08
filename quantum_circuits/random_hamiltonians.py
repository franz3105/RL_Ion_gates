# -*- coding: utf-8 -*-

from numpy.linalg import norm
from quantum_circuits.exponentials import *

# A file containing functions to generate random Hamiltonians and wave functions

def randH(s, amp=1, sparsity=0.95, rng=random.default_rng()):

    """
    Generate a random Hermitian matrix.
    :param s: Size of the matrix.
    :param amp: Amplitude of the random matrix.
    :param sparsity: Sparsity of the random matrix.
    :param rng: Random number generator.
    :return: Random Hermitian matrix (numpy ndarray).
    """
    s = int(round(s))

    H = rng.random((s, s)) + 1j * rng.random((s, s))

    H = (H + transpose(conj(H))) / 2

    if sparsity > 0:

        c = random.rand(s, s)

        c = (c + transpose(conj(c))) / 2

        d = where(c < sparsity)

        for i in range(d[0].size):
            H[d[0][i], d[1][i]] = 0

    # Normalize

    Hnorm = norm(H)  # Hilbert Schmidt Norm

    if Hnorm > 1e-14:
        H = H / Hnorm * amp
    print(H)
    return H


def randU(s, amp=1, sparsity=0.95, rng=random.default_rng()):
    """
    Generate a random unitary matrix.
    :param s: Size of the matrix.
    :param amp: Amplitude of the random matrix.
    :param sparsity: Sparsity of the random matrix.
    :param rng: Random number generator.
    :return: Random unitary matrix (numpy ndarray).
    """
    return expmH(randH(s, amp, sparsity, rng), pi)


def Rand_wf(s, how_many=1, rng=random.default_rng()):

    """
    Generate a random wave function.
    :param s: Size of the vector.
    :param how_many: Number of random vectors.
    :param rng: Random number generator.
    :return: Random wave function (numpy ndarray).
    """
    wf = rng.random((s, how_many))

    for i in range(how_many):
        wf[:, i] = wf[:, i] / norm(wf[:, i])

    phi = 2 * pi * rng.random((s, how_many))

    wf = multiply(wf, exp(-1j * phi))

    return wf
