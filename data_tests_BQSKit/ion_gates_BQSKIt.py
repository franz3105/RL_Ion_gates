"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitary import RealVector
from bqskit.utils.cachedclass import CachedClass
from quantum_circuits.exponentials import expmH
from quantum_circuits.quantum_gates_numba import pauli_ops
from scipy.linalg import expm

class MSGate(
    QubitGate,
    DifferentiableUnitary
):
    _num_qudits = 3
    _num_params = 2
    _qasm_name = 'msgate'

    def __init__(self, num_qudits):

        self.Sx, self.Sy = pauli_ops(num_qudits)
        self.Sx2 = self.Sx.dot(self.Sx)
        self.Sy2 = self.Sy.dot(self.Sy)
        self.SxSy = self.Sx.dot(self.Sy) + self.Sy.dot(self.Sx)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        # self.check_parameters(params)
        h = self.Sx2 * np.cos(params[1]) ** 2 + self.Sy2 * np.sin(params[1]) ** 2 +\
            self.SxSy * np.cos(params[1]) * np.sin(params[1])

        out = - 1.j * params[0] * h / 4
        exp_out = expm(out)
        # print(exp_out.conj().T.dot(exp_out))

        return UnitaryMatrix(exp_out)

    def get_grad(self, params: RealVector = []) -> np.ndarray:
        """
    #    Return the gradient for this gate.

    #    See :class:`DifferentiableUnitary` for more info.
    #    """
        self.check_parameters(params)

        h = self.Sx2 * np.cos(params[1]) ** 2 + self.Sy2 * np.sin(params[1]) ** 2 + \
            self.SxSy * np.cos(params[1]) * np.sin(params[1])
        out = - 1.j * params[0] * h / 4
        exp_theta_dev = - 1.j * h / 4
        exp_out = expm(out)
        shifted_h = self.Sx2 * np.cos(params[1] + 1e-8) ** 2 + self.Sy2 * np.sin(params[1] + 1e-8) ** 2 + \
                    self.SxSy * np.cos(params[1] + 1e-8) * np.sin(params[1] + 1e-8)
        shifted_exp_out = expm(- 1j * params[0] * shifted_h / 4)

        return np.array([

            exp_theta_dev.dot(exp_out),
            (shifted_exp_out - exp_out) / 1e-8

        ]
            , dtype=np.complex128,
        )


class XYGate(
    QubitGate,
    DifferentiableUnitary
):
    _num_qudits = 3
    _num_params = 2
    _qasm_name = 'xygate'

    def __init__(self, num_qudits):
        self.Sx, self.Sy = pauli_ops(num_qudits)
        #self.Sx2 = self.Sx.dot(self.Sx)
        #self.Sy2 = self.Sy.dot(self.Sy)
        #self.SxSy = self.Sx.dot(self.Sy) + self.Sy.dot(self.Sx)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        # self.check_parameters(params)
        out = - 1.j * params[0] * (self.Sx * np.cos(params[1]) + self.Sy * np.sin(params[1])) / 2
        exp_out = expm(out)
        # print(exp_out.conj().T.dot(exp_out))

        return UnitaryMatrix(exp_out)

    def get_grad(self, params: RealVector = []) -> np.ndarray:
        """
    #    Return the gradient for this gate.

    #    See :class:`DifferentiableUnitary` for more info.
    #    """
        self.check_parameters(params)

        out = - 1.j * params[0] * (self.Sx * np.cos(params[1]) + self.Sy * np.sin(params[1])) / 2
        exp_out = expm(out)
        shifted_out = - 1.j * params[0] * (self.Sx * np.cos(params[1] + 1e-8) + self.Sy * np.sin(params[1] + 1e-8)) / 2
        shifted_exp_out = expm(shifted_out)

        exp_theta_dev = - 1.j * (self.Sx * np.cos(params[1]) + self.Sy * np.sin(params[1])) / 2

        return np.array([

            exp_theta_dev.dot(exp_out),
            (shifted_exp_out - exp_out) / 1e-8

        ]
            , dtype=np.complex128,
        )