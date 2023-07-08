import numpy as np

from numba import njit, jit
from numba.typed import List
from typing import List, Tuple, Dict, Callable
# Simple quantum gates without spectral decomposition

sz = np.array([[1, 0], [0, -1]], dtype=np.complex128, order="C")
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128, order="C")
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128, order="C")


@njit(fastmath=True, nogil=True)
def tensor(op_list: List):
    """
     Returns the tensor product of a list of quantum gates.

     Parameters
     ----------
     op_list : List
        A list of quantum gates.

     Returns
     -------
     out : np.ndarray
         The tensor product of the quantum gates.
     """
    s = op_list[0]
    for i_op in range(1, len(op_list)):
        s = np.kron(s, op_list[i_op])

    return s


@njit(fastmath=True, nogil=True)
def Dag(U):
    """
     Returns the adjoint of a quantum gate.

     Parameters
     ----------
     U : np.ndarray
        A quantum gate.

     Returns
     -------
     out : np.ndarray
         The adjoint of the quantum gate.
     """
    U_dag = np.conjugate(U).T
    return U_dag


# Construct matrix exponentials and products faster than in numpy or scipy
@njit(fastmath=True, nogil=True)
def vm_exp_mul(E, V, dt=1.0):  # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    """
    Returns the matrix exponential of a vector.

    Parameters
    ----------
    E : np.ndarray
        The vector.
    V : np.ndarray
        The matrix.
    dt : float, optional
        The time step. The default is 1.0.

    Returns
    -------
    out : np.ndarray
        The matrix exponential of the vector.
    """
    s = E.size
    A = np.empty((s, s), np.complex128)
    for k in range(s):
        C = -1j * E[k] * dt
        A[k, :] = np.exp(C) * Dag(V[:, k])

    return A


@njit(fastmath=True)
def expmH_from_Eig(E, V, dt=1.0):
    """
    Returns the matrix exponential of a vector given eigenvalues and eigenvectors.
    :param E: Eigenvalues of the matrix.
    :param V: Eigenvectors of the matrix.
    :param dt: The time step.
    :return: The matrix exponential of the vector.
    """
    U = np.dot(V, vm_exp_mul(E, V, dt))
    return U


@njit(fastmath=True)
def get_eigh(H):
    """
    Returns the eigenvalues and eigenvectors of a matrix.

    Parameters
    ----------
    H : np.ndarray
        The matrix.

    Returns
    -------
    out : np.ndarray
        The eigenvalues and eigenvectors of the matrix.
    """
    return np.linalg.eigh(1j * H)


@njit(fastmath=True)
def expmH(H, dt=1):

    """
    Exponential matrix through spectral decomposition.
    :param H: Hamiltonian.
    :param dt: The time step.
    :return: The matrix exponential of the Hamiltonian.
    """

    E, V = get_eigh(H)
    U = expmH_from_Eig(E, V, dt)
    return U


@njit(fastmath=True)
def basis(d: int, idx: int) -> np.ndarray:

    """
    Defines one of the standard basis vectors of a Hilbert space with index idx.
    :param d: The dimension of the Hilbert space.
    :param idx: The index of the basis element.
    :return: The basis element (quantum state).
    """

    zero_tensor = np.zeros([d, 1])
    zero_tensor[idx, :] = 1

    return zero_tensor


def pauli_ops(n_qudits):

    """
    Returns a list of Pauli operators.
    :param n_qudits: The number of qudits.
    :return: A list of Pauli operators.
    """
    Sx = np.zeros((2 ** n_qudits, 2 ** n_qudits), dtype=np.complex128)
    Sy = np.zeros((2 ** n_qudits, 2 ** n_qudits), dtype=np.complex128)

    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    for k in range(n_qudits):
        qubit_list[k] = sx
        Sx += tensor(qubit_list)
        qubit_list[k] = sy
        Sy += tensor(qubit_list)
        qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits

    return Sx, Sy


def pauli_ops_i_j(i, j, n_qudits):
    """
    Returns a list of Pauli operators.
    :param i: The index of the first qubit.
    :param j: The index of the second qubit.
    :param n_qudits: The number of qudits.
    :return: A list of Pauli operators.
    """
    Sx = np.zeros([2 ** n_qudits, 2 ** n_qudits], dtype=np.complex128)
    Sy = np.zeros([2 ** n_qudits, 2 ** n_qudits], dtype=np.complex128)

    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    for i in range(i, j):
        qubit_list[i] = sx
        Sx += tensor(qubit_list)
        qubit_list[i] = sy
        Sy += tensor(qubit_list)
        qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits

    return Sx, Sy


def qudit_pauli_ops(d: int, num_qudits: int) -> List:

    """
    Returns a list of Pauli operators.
    :param d: The dimension of the Hilbert space.
    :param num_qudits: The number of qudits.
    :return: A list of Pauli operators.
    """

    if d % 2 == 0:
        raise ValueError("Only odd numbers allowed")

    s = int((d - 1) / 2)
    S = basis(d, 0).dot(basis(d, 0).T.conjugate()) - basis(d, 0).dot(basis(d, 0).T.conjugate())
    for l in range(-s, s):
        S += np.sqrt(s * (s + 1) - l * (l + 1)) * (basis(d, l + s + 1).dot(basis(d, l + s).T.conjugate())
                                                   + basis(d, l + s).dot(basis(d, l + s + 1).T.conjugate()))
    # print(S)
    qudit_list = [np.eye(d, dtype=np.complex128)] * num_qudits
    sum_op = tensor([np.zeros([d, d])] * num_qudits)
    for i in range(num_qudits):
        qudit_list[i] = S
        sum_op += tensor(qudit_list)
        qudit_list = [np.zeros([d, d])] * num_qudits

    return sum_op


@njit(fastmath=True)
def Cxy(theta: np.float64, phi: np.float64, Sx: np.ndarray, Sy: np.ndarray) -> np.ndarray:

    """
    The XY gate unitary compute through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param phi: The phase of the rotation.
    :param Sx: Sum of all sigma_x matrices acting on qubits 1 to n.
    :param Sy: Sum of all sigma_y matrices acting on qubits 1 to n.
    :return: The XY gate unitary.
    """
    out = - 1.j * theta * (Sx * np.cos(phi) + Sy * np.sin(phi)) / 2
    exp_out = expmH(out)
    # print(expmH(out))
    # print(sp.linalg.expm(out))
    # print(np.array_equal(np.round(expmH(out),5), np.round(sp.linalg.expm(out),5)))
    return exp_out


@njit(fastmath=True)
def Zloc(theta: np.float64, k: int, n_qudits: int):

    """
    The Z gate unitary compute through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param k: The index of the qudit.
    :param n_qudits: The number of qudits.
    :return: The Z gate unitary.
    """

    out = - 1.j * theta * sz
    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    # print(out)
    qubit_list[k] = expmH(out)
    # print(expmH(out))
    # print(sp.linalg.expm(out))
    # print(np.array_equal(np.round(expmH(out),5), np.round(sp.linalg.expm(out),5)))
    out = tensor(qubit_list)
    return out


@njit(fastmath=True)
def Zloc_and_grad(theta: np.float64, k: int, n_qudits: int) -> Tuple[np.ndarray, np.ndarray]:

    """
    The Z gate unitary and its gradients compute through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param k: The index of the qudit.
    :param n_qudits: The number of qudits.
    :return: The Z gate unitary.
    """
    out = - 1.j * theta * sz / 2
    grad_factor = - 1.j * sz / 2
    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    qubit_list_2 = [np.eye(2, dtype=np.complex128)] * n_qudits

    # print(out)
    qubit_list[k] = expmH(out)
    qubit_list_2[k] = grad_factor.dot(qubit_list[k])
    # print(expmH(out))
    # print(sp.linalg.expm(out))
    # print(np.array_equal(np.round(expmH(out),5), np.round(sp.linalg.expm(out),5)))
    out = tensor(qubit_list)
    out_2 = tensor(qubit_list)
    return out, out_2


# @njit(fastmath=True )

def Xloc(theta: np.float64, i: int, n_qudits: int):

    """
    The X gate unitary and its gradients compute through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param k: The index of the qudit.
    :param n_qudits: The number of qudits.
    :return: The X gate unitary.
    """
    out = - 1.j * theta * sx / 2
    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    qubit_list[i] = expmH(out)
    out = tensor(qubit_list)
    return out


def Yloc(theta: np.float64, i: int, n_qudits: int):

    """
    The Y gate unitary and its gradients compute through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param k: The index of the qudit.
    :param n_qudits: The number of qudits.
    :return: The local Y gate unitary.
    """
    out = - 1.j * theta * sy / 2
    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    qubit_list[i] = expmH(out)
    out = tensor(qubit_list)
    return out


def R_xyz(alpha, beta, gamma, index):
    """
    The XY gate times a local Z gate unitary.
    :param alpha:   The angle of the rotation of XY.
    :param beta:    The phase of the rotation of XY.
    :param gamma:   The index of the qudit.
    :param index:   The index of the qudit.
    :return: The corresponding gate
    """
    return Cxy(alpha, beta).dot(Zloc(gamma, index))


@njit(fastmath=True)
def MS(theta, phi, Sx2, Sy2, SxSy):

    """
    The Moelmer Soerensen computed through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param phi: The phase of the rotation.
    :param Sx2: Sum of all sigma_x matrices acting on qubits 1 to n.
    :param Sy2: Sum of all sigma_y matrices acting on qubits 1 to n.
    :param SxSy: Sum of all sigma_xy matrices acting on qubits 1 to n.
    :return: The Moelmer Soerensen.
    """
    h = Sx2 * np.cos(phi) ** 2 + Sy2 * np.sin(phi) ** 2 + SxSy * np.cos(phi) * np.sin(phi)

    out = - 1.j * theta * h / 4
    exp_out = expmH(out)
    # print(expmH(out))
    # print(sp.linalg.expm(out))
    # print(np.array_equal(np.round(expmH(out),5), np.round(sp.linalg.expm(out),5)))
    return exp_out


def MS2q(theta, phi, Sx2q, Sy2q):

    """
    The Moelmer Soerensen for 2 qubtis computed through matrix exponentiation.
    :param theta: The angle of the rotation.
    :param phi: The phase of the rotation.
    :param Sx2q: Sum of all sigma_x matrices acting on qubits 1 to n.
    :param Sy2q: Sum of all sigma_y matrices acting on qubits 1 to n.
    :return: The Moelmer Soerensen.
    """
    mix = (Sx2q * np.cos(phi) + Sy2q * np.sin(phi))
    out = - 1.j * theta * mix.dot(mix) / 4
    out = expmH(out)
    return out


def MSnq(theta, phi, idx_hash, n_qudits):

    """

    :param theta: The angle of the rotation.
    :param phi: The phase of the rotation.
    :param idx_hash: The index of the qudit.
    :param n_qudits: The number of qudits.
    :return: The Moelmer Soerensen.
    """
    Sx2q = np.zeros([2 ** n_qudits, 2 ** n_qudits], dtype=np.complex128)
    Sy2q = np.zeros([2 ** n_qudits, 2 ** n_qudits], dtype=np.complex128)

    qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
    for idx in idx_hash:
        try:
            qubit_list[idx] = sx
            Sx2q += tensor(qubit_list)
            qubit_list[idx] = sy
            Sy2q += tensor(qubit_list)
            qubit_list = [np.eye(2, dtype=np.complex128)] * n_qudits
        except IndexError:
            raise IndexError(f"Indices (i_1, ..., i_r) ={str(idx_hash)} must lie between 0 and n_qudits={n_qudits}")

    mix = (Sx2q * np.cos(phi) + Sy2q * np.sin(phi))
    out = - 1.j * theta * mix.dot(mix) / 4
    out = expmH(out)
    return out


def MSG(theta, sum_op2):  # This gate is still missing a phase

    out = - 1.j * theta * sum_op2 / 4
    out = expmH(out)
    # print(out)
    return out


def Ry(theta, i, j, k, n_qudits, d):

    """
    Computes the qudit rotation of the local Y gate (in the qudit sense).
    :param theta: The angle of the rotation.
    :param i: The index of the qudit.
    :param j: The index of the qudit.
    :param k: The index of the qudit.
    :param n_qudits: The number of qudits.
    :param d: The index of the qudit.
    :return: The qudit rotation of the local Y gate.
    """
    sigmay01 = -1j * basis(d, j).dot(basis(d, k).T.conjugate()) + 1j * basis(d, k).dot(basis(d,
                                                                                             j).T.conjugate())
    out = -1j * theta * sigmay01 * 0.5
    qudit_list = [np.eye(d, dtype=np.complex128)] * n_qudits
    Sy = tensor([np.zeros([d, d])] * n_qudits)
    qudit_list[i] = sigmay01
    Sy += tensor(qudit_list)
    out = - 1.j * theta * Sy / 2
    out = expmH(out)
    return out


def Rz(theta: np.float64, i: int, j: int, k: int, n_qudits: int, d: int) -> np.ndarray:

    """
    Computes the qudit rotation of the global Z gate (in the qudit sense).
    :param theta:
    :param i: The index of the qudit.
    :param j: The index of the first qudit level.
    :param k: The index of the second qudit level.
    :param n_qudits: Number of qudits.
    :param d: Dimension of the qudits space.
    :return: Generalized multi-qudit-Z interaction.
    """
    sigmaz01 = basis(d, j).dot(basis(d, j).T.conjugate()) - basis(d, k).dot(basis(d, k).T.conjugate())
    out = -1j * theta * sigmaz01 * 0.5
    qudit_list = [np.eye(d, dtype=np.complex128)] * n_qudits
    Sy = tensor([np.zeros([d, d])] * n_qudits)
    qudit_list[i] = sigmaz01
    Sy += tensor(qudit_list)
    out = - 1.j * theta * Sy / 2
    out = expmH(out)
    return out


def RGy(theta, j, k, n_qudits, d):

    """
    Computes the qudit rotation of the global Y gate (in the qudit sense (- ik + kj)  for 0<k<j<d-1).
    :param theta:
    :param i: The index of the qudit.
    :param j: The index of the first qudit level.
    :param k: The index of the second qudit level.
    :param n_qudits: Number of qudits.
    :param d: Dimension of the qudits space.
    :return: Generalized qudit-Y interaction.
    """
    assert d - 1 > j >= 0
    assert d - 1 > k >= 0
    sigmay = -1j * basis(d, j).dot(basis(d, k).T.conjugate()) + 1j * basis(d, k).dot(basis(d,
                                                                                           j).T.conjugate())
    qudit_list = [np.eye(d, dtype=np.complex128)] * n_qudits
    Sy = tensor([np.zeros([d, d])] * n_qudits)
    for i in range(n_qudits):
        qudit_list[i] = sigmay
        Sy += tensor(qudit_list)
        qudit_list = [np.eye(d, dtype=np.complex128)] * n_qudits
    out = - 1.j * theta * Sy / 2
    out = expmH(out)
    return out


def RGx(theta, j, k, n_qudits, d):

    """
    Computes the qudit rotation of the global X gate (in the qudit sense).
    :param theta:
    :param i: The index of the qudit.
    :param j: The index of the first qudit level.
    :param k: The index of the second qudit level.
    :param n_qudits: Number of qudits.
    :param d: Dimension of the qudits space.
    :return: Generalized multi-qudit-X interaction.
    """
    sigmax = basis(d, j).dot(basis(d, k).T.conjugate()) + basis(d, k).dot(basis(d, j).T.conjugate())
    qudit_list = [np.eye(d, dtype=np.complex128)] * n_qudits
    Sx = tensor([np.zeros([d, d])] * n_qudits)
    for i in range(n_qudits):
        qudit_list[i] = sigmax
        Sx += tensor(qudit_list)
        qudit_list = [np.eye(d, dtype=np.complex128)] * n_qudits
    out = - 1.j * theta * Sx / 2
    out = expmH(out)
    return out


def grad_theta_MS2q(theta, phi, Sx2q, Sy2q):

    """
    Derivative with respect to theta of the 2-qubit MS gate.
    :param theta: Rotation angle of the 2-qubit MS gate.
    :param phi: Phase angle of the 2-qubit MS gate.
    :param Sx2q: two-qubit XX operator.
    :param Sy2q: two-qubit YY operator.
    :return: The derivative of the 2-qubit MS gate with respect to theta.
    """
    exp_theta_dev = - 1.j * (Sx2q * np.cos(phi) + Sy2q * np.sin(phi)).dot(
        (Sx2q * np.cos(phi) + Sy2q * np.sin(phi))) / 4
    return MS2q(theta, phi, Sx2q, Sy2q).dot(exp_theta_dev)


def grad_phi_MS2q(theta, phi, Sx2q, Sy2q):

    """
    Numerical derivative with respect to phi of the 2-qubit MS gate.
    :param theta: Rotation angle of the 2-qubit MS gate.
    :param phi: Phase angle of the 2-qubit MS gate.
    :param Sx2q: two-qubit XX operator.
    :param Sy2q: two-qubit YY operator.
    :return: The numerical derivative of the 2-qubit MS gate with respect to phi.
    """
    phi_new = phi + 1e-7

    return (MS2q(theta, phi_new, Sx2q, Sy2q) - MS2q(theta, phi, Sx2q, Sy2q)) * 1e7


@njit(fastmath=True)
def grad_phi_MS(theta, phi, Sx2, Sy2, SxSy):

    """
    Numerical Derivative with respect to phi of the n-qubit MS gate.
    :param theta: Rotation angle of the n-qubit MS gate.
    :param phi: Phase angle of the n-qubit MS gate.
    :param Sx2: n-qubit XX...X operator.
    :param Sy2: n-qubit YY...Y operator.
    :param SxSy: n-qubit mixed XX...Y operator.
    :return: The numerical derivative of the n-qubit MS gate with respect to phi.
    """
    phi_new = phi + 1e-7

    return (MS(theta, phi_new, Sx2, Sy2, SxSy) - MS(theta, phi, Sx2, Sy2, SxSy)) * 1e7


@njit(fastmath=True)
def grad_theta_MS(theta, phi, Sx2, Sy2, SxSy):

    """
    Derivative with respect to theta of the n-qubit MS gate.
    :param theta: Rotation angle of the n-qubit MS gate.
    :param phi: Phase angle of the n-qubit MS gate.
    :param Sx2: n-qubit XX...X operator.
    :param Sy2: n-qubit YY...Y operator.
    :param SxSy: n-qubit mixed XX...Y operator.
    :return: The numerical derivative of the n-qubit MS gate with respect to theta.
    """
    h = Sx2 * np.cos(phi) ** 2 + Sy2 * np.sin(phi) ** 2 + SxSy * np.cos(phi) * np.sin(phi)

    exp_theta_dev = - 1.j * h / 4

    return MS(theta, phi, Sx2, Sy2, SxSy).dot(exp_theta_dev)


@njit(fastmath=True)
def grad_theta_Cxy(theta, phi, Sx, Sy):

    """
    Derivative with respect to theta of the n-qubit XY gate.
    :param theta: Rotation angle of the n-qubit MS gate.
    :param phi: Phase angle of the n-qubit MS gate.
    :param Sx: n-qubit XX...X operator.
    :param Sy: n-qubit YY...Y operator.
    :return: The numerical derivative of the n-qubit XY gate with respect to theta.
    """
    exp_theta_dev = - 1.j * (Sx * np.cos(phi) + Sy * np.sin(phi)) / 2
    return Cxy(theta, phi, Sx, Sy).dot(exp_theta_dev)


@njit(fastmath=True)
def grad_phi_Cxy(theta, phi, Sx, Sy):
    """
    Numerical derivative with respect to phi of the n-qubit XY gate.
    :param theta: Rotation angle of the n-qubit MS gate.
    :param phi: Phase angle of the n-qubit MS gate.
    :param Sx: n-qubit XX...X operator.
    :param Sy: n-qubit YY...Y operator.
    :return: The numerical derivative of the n-qubit XY gate with respect to phi.
    """
    phi_plus = phi + 1e-7
    phi_minus = phi - 1e-7
    num_grad = (Cxy(theta, phi_plus, Sx, Sy) - Cxy(theta, phi_minus, Sx, Sy)) / (2 * 1e-7)
    # anal_grad = Cxy(theta, phi, Sx, Sy).dot(- 1.j * theta / 2 * (-Sx * np.sin(phi) + Sy * np.cos(phi)))
    # print(np.max(num_grad - anal_grad))
    return num_grad


@njit(fastmath=True)
def grad_theta_Zloc(theta, i, num_qubits):

    """
    Derivative with respect to theta of the single qubit Zloc gate.
    :param theta: Rotation angle of the single qubit Zloc gate.
    :param i: The index of the qubit.
    :param num_qubits:  Number of qubits.
    :return: The derivative of the single qubit Zloc gate with respect to theta.
    """
    out = - 1.j * theta * sz / 2
    qubit_list = [np.eye(2, dtype=np.complex128)] * num_qubits
    qubit_list[i] = expmH(out).dot(-1j * sz / 2)
    out = tensor(qubit_list)
    return out


@njit(fastmath=True)
def grad_theta_Xloc(theta, i, num_qubits):

    """
    Derivative with respect to theta of the single qubit Xloc gate.
    :param theta: Rotation angle of the single qubit Xloc gate.
    :param i:   The index of the qubit.
    :param num_qubits: The number of qubits.
    :return:
    """
    out = - 1.j * theta * sx / 2
    qubit_list = [np.eye(2, dtype=np.complex128)] * num_qubits
    qubit_list[i] = expmH(out).dot(-1j * sz / 2)
    out = tensor(qubit_list)
    return out


@njit(fastmath=True)
def grad_theta_Yloc(theta, i, num_qubits):

    """
    Derivative with respect to theta of the single qubit Yloc gate.
    :param theta: Rotation angle of the single qubit Yloc gate.
    :param i:  The index of the qubit.
    :param num_qubits: The number of qubits.
    :return: The derivative of the single qubit Yloc gate with respect to theta.
    """
    out = - 1.j * theta * sy / 2
    qubit_list = [np.eye(2, dtype=np.complex128)] * num_qubits
    qubit_list[i] = expmH(out).dot(-1j * sz / 2)
    out = tensor(qubit_list)
    return out


def create_standard_ms_set(num_qubits):

    """
    Creates the standard set of gates for the MS gate with the not-so-fast gates.
    :param num_qubits: the number of qubits
    :return:    the set of gates
    """
    Sx, Sy = pauli_ops(num_qubits)
    Sx2 = Sx.dot(Sx)
    Sy2 = Sy.dot(Sy)
    SxSy = Sx.dot(Sy) + Sy.dot(Sx)

    @njit(fastmath=True)
    def ms_gate(theta, phi):
        return MS(theta, phi, Sx2, Sy2, SxSy)

    # @timing
    @njit(fastmath=True)
    def cxy_gate(theta, phi):
        return Cxy(theta, phi, Sx, Sy)

    def get_zloc(i):
        @njit(fastmath=True)
        def z_iqubit(theta):
            return Zloc(theta, i, n_qudits=num_qubits)

        return z_iqubit

    gate_set = [ms_gate, cxy_gate] + [get_zloc(k) for k in range(num_qubits)]
    gate_names = ["MS", "Cxy"] + [f"Zloc_{k + 1}" for k in range(num_qubits)]

    @njit(fastmath=True)
    def ms_grad_theta(theta, phi):
        return grad_theta_MS(theta, phi, Sx2, Sy2, SxSy)

    @njit(fastmath=True)
    def ms_grad_phi(theta, phi):
        return grad_phi_MS(theta, phi, Sx2, Sy2, SxSy)

    @njit(fastmath=True)
    def cxy_grad_theta(theta, phi):
        return grad_theta_Cxy(theta, phi, Sx, Sy)

    @njit(fastmath=True)
    def cxy_grad_phi(theta, phi):
        return grad_phi_Cxy(theta, phi, Sx, Sy)

    def get_z_loc_grad(i):
        @njit(fastmath=True)
        def f_z_grad(theta):
            return grad_theta_Zloc(theta, i, num_qubits=num_qubits)

        return f_z_grad

    gate_gradients = dict(MS=(ms_grad_theta, ms_grad_phi), Cxy=(cxy_grad_theta, cxy_grad_phi))

    for n in range(num_qubits):
        gate_gradients[f"Zloc_{n + 1}"] = get_z_loc_grad(n)

    gate_set = gate_set
    gate_names = gate_names

    return gate_set, gate_names, gate_gradients


def create_standard_ms_set_2(num_qubits: int) ->    Tuple[List[Callable], List[str], Dict[str, Callable]]:

    """
    Creates the standard set of gates for the MS gate with the not-so-fast gates.
    :param num_qubits: The number of qubits.
    :return: Gate functions, gate names, gradients.
    """
    Sx, Sy = pauli_ops(num_qubits)
    Sx2 = Sx.dot(Sx)
    Sy2 = Sy.dot(Sy)
    SxSy = Sx.dot(Sy) + Sy.dot(Sx)

    # @njit(fastmath=True )
    def ms_gate(theta, phi):
        return MS(theta, phi, Sx2, Sy2, SxSy)

    # @timing
    # @njit(fastmath=True )
    def cxy_gate(theta, phi):
        return Cxy(theta, phi, Sx, Sy)

    def get_zloc():
        # @njit(fastmath=True )
        def z_iqubit(theta, i):
            return Zloc(theta, i, n_qudits=num_qubits)

        return z_iqubit

    gate_set = [ms_gate, cxy_gate] + [get_zloc()]
    gate_names = ["MS", "Cxy"] + ["Zloc"]

    # @njit(fastmath=True )
    def ms_grad_theta(theta, phi):
        return grad_theta_MS(theta, phi, Sx2, Sy2, SxSy)

    # @njit(fastmath=True )
    def ms_grad_phi(theta, phi):
        return grad_phi_MS(theta, phi, Sx2, Sy2, SxSy)

    # @njit(fastmath=True )
    def cxy_grad_theta(theta, phi):
        return grad_theta_Cxy(theta, phi, Sx, Sy)

    # @njit(fastmath=True )
    def cxy_grad_phi(theta, phi):
        return grad_phi_Cxy(theta, phi, Sx, Sy)

    def get_z_loc_grad():
        # @njit(fastmath=True )
        def f_z_grad(theta, i):
            return grad_theta_Zloc(theta, i, num_qubits=num_qubits)

        return f_z_grad

    gate_gradients = dict(MS=(ms_grad_theta, ms_grad_phi), Cxy=(cxy_grad_theta, cxy_grad_phi))

    # for n in range(num_gates):
    gate_gradients[f"Zloc"] = get_z_loc_grad()

    gate_set = gate_set
    gate_names = gate_names

    return gate_set, gate_names, gate_gradients


def create_2q_ms_set(qudit_dim, num_qubits):

    """
    Creates the two-qubits set of gates for the MS gate with the not-so-fast gates.
    :param qudit_dim: the dimension of the qudits.
    :param num_qubits: the number of qubits
    :return: the set of gates
    """
    gate_gradients = dict()

    def create_ms_2q_gate(k, j):
        Sx2q, Sy2q = pauli_ops_i_j(k, j, n_qudits=num_qubits)

        def ms_2q_ij(theta, phi):
            return MS2q(theta, phi, Sx2q, Sy2q)

        return ms_2q_ij

    def create_zloc(k):
        def z_iqubit(theta):
            return Zloc(theta, k, n_qudits=num_qubits)

        return z_iqubit

    def create_xloc(k):
        def x_1qubit(theta):
            return Xloc(theta, k, n_qudits=num_qubits)

        return x_1qubit

    def create_yloc(k):
        def y_1qubit(theta):
            return Yloc(theta, k, n_qudits=num_qubits)

        return y_1qubit

    gate_set = [create_ms_2q_gate(k, j) for k in range(num_qubits) for j in range(k)] + \
               [create_zloc(k) for k in range(num_qubits)] + \
               [create_yloc(k) for k in range(num_qubits)] + \
               [create_xloc(k) for k in range(num_qubits)]

    def create_ms2q_grad_theta(k, j):
        Sx2q, Sy2q = pauli_ops_i_j(k, j, n_qudits=num_qubits)

        def ms_2q_grad_theta_ij(theta, phi):
            return grad_theta_MS2q(theta, phi, Sx2q, Sy2q)

        return ms_2q_grad_theta_ij

    def create_ms2q_grad_phi(k, j):
        Sx2q, Sy2q = pauli_ops_i_j(k, j, n_qudits=num_qubits)

        def ms_2q_grad_phi_ij(theta, phi):
            return grad_phi_MS2q(theta, phi, Sx2q, Sy2q)

        return ms_2q_grad_phi_ij

    def create_zloc_grad(k):
        def z_1qubit_grad(theta):
            return grad_theta_Zloc(theta, k, n_qudits=num_qubits)

        return z_1qubit_grad

    def create_xloc_grad(k):
        def x_1qubit_grad(theta):
            return grad_theta_Xloc(theta, k, n_qudits=num_qubits)

        return x_1qubit_grad

    def create_yloc_grad(k):
        def y_1qubit_grad(theta):
            return grad_theta_Yloc(theta, k, n_qudits=num_qubits)

        return y_1qubit_grad

    for n in range(num_qubits):
        gate_gradients[f"Zloc_{n + 1}"] = create_zloc_grad(n)
        gate_gradients[f"Xloc_{n + 1}"] = create_xloc_grad(n)
        gate_gradients[f"Yloc_{n + 1}"] = create_yloc_grad(n)
        for j in range(n):
            gate_gradients[f"MS_{j}_{n}"] = (create_ms2q_grad_theta(n, j), create_ms2q_grad_phi(n, j))

    gate_set = tuple(gate_set)

    return gate_set, gate_gradients


def create_qudit_ms_set(qudit_dim, num_qudits):
    """
    Creates the gate set for qudits.
    :param qudit_dim:
    :param num_qudits:
    :return:
    """
    gate_set = []
    gate_gradients = dict()

    def ms_qudit_gate(theta, n_qudits, d):
        sum_op2 = qudit_pauli_ops(n_qudits, d)
        sum_op2 = sum_op2.dot(sum_op2)
        return MSG(theta, sum_op2)

    def rgx_qudit_gate(theta, n_qudits, d):
        sum_op2 = qudit_pauli_ops(n_qudits, d)
        return RGx(theta, sum_op2)

    def rgy_qudit_gate(theta, n_qudits, d):
        sum_op2 = qudit_pauli_ops(n_qudits, d)
        return RGy(theta, sum_op2)

    gate_set = tuple(gate_set)

    return gate_set, gate_gradients
