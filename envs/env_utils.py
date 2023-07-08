# -*- coding: utf-8 -*-
import jax
import qutip as qt
import numpy as np

from functools import partial
from quantum_circuits.cost_and_grad_numba import create_cost_gates_standard, create_cost_states_standard
from quantum_circuits.cost_and_grad_jax import create_simple_cost_gates, pvqd_evo_cost
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from quantum_circuits.gate_set_jax import create_fast_ms_set_jax
from quantum_circuits.layer_env_cost_grad import create_cost_gates_layers
from jax import jit
from jax.lib import xla_bridge


def construct_cost_function(gate_set_type: str, library: str, num_qubits: int, target: np.ndarray
                            , max_iter=100, device=0, time_dep_u=True):
    """
    Construct the cost function for the gate design problem.
    :param device: Device (in case of jax) where the cost function will be executed.
    :param max_iter: Maximum number of iterations for the optimization algorithm.
    :param gate_set_type: Type of gate set to use.
    :param library: Library to use for the cost function.
    :param num_qubits: Number of qubits.
    :param target: Target gate.
    :param time_dep_u: Whether to use time dependent unitaries.
    :return: gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt
    """
    print("Constructing cost function...")
    if len(target.shape) == 2 and target.shape[1] == target.shape[0]:
        if library == "numba":
            if gate_set_type == "standard":
                gate_funcs, gate_names, structure = create_fast_ms_set_numba(num_qubits, z_prod=False)
                cs_to_unitaries, cost_grad = create_cost_gates_standard(target, *gate_funcs)
                vec_cost_grad = cost_grad
                ad_opt_alg = None
            elif gate_set_type == "layers":
                gate_funcs, gate_names, structure = create_fast_ms_set_numba(num_qubits, z_prod=True)
                cs_to_unitaries, cost_grad = create_cost_gates_layers(num_qubits, target, *gate_funcs)
                vec_cost_grad = cost_grad
                ad_opt_alg = None
            else:
                raise ValueError("Gate set type not recognized.")

        elif library == "jax":
            gate_funcs, gate_names, _ = create_fast_ms_set_jax(num_qubits)
            cs_to_unitaries, cost_grad, vec_cost_grad, ad_opt_alg = pvqd_evo_cost(target, gate_funcs, max_iter=max_iter,
                                                                                  n_devices=1)
            platform = xla_bridge.get_backend().platform

            if "gpu" in platform:
                gpus = jax.devices("gpu")
                print(gpus)
                assert device < len(gpus)
                vec_cost_grad = jit(vec_cost_grad, device=gpus[device])
                ad_opt_alg = partial(jit, static_argnums=(2,), device=gpus[device])(ad_opt_alg)
        else:
            raise ValueError("Library not supported.")

    elif len(target.shape) == 1:
        if library == "numba":
            gate_funcs, gate_names, structure = create_fast_ms_set_numba(num_qubits, z_prod=False)
            cs_to_unitaries, cost_grad = create_cost_states_standard(num_qubits, target[:, 1], target[:, 0], *gate_funcs)
            vec_cost_grad = cost_grad
            ad_opt_alg = None
        else:
            raise ValueError("Library not supported for states.")
    else:
        raise ValueError("Library not supported.")

    #print(cs_to_unitaries)
    return gate_funcs, gate_names, cost_grad, vec_cost_grad, ad_opt_alg, cs_to_unitaries


def map_list(func_list, object_list):

    """

    :param func_list:
    :param object_list:
    :return:
    """

    s = []
    for element in func_list:
        s += list(map(element, object_list))
    return s


def get_names(func_list):

    """

    :param func_list:
    :return:
    """

    s = []
    for element in func_list:
        s.append(repr(element))
    return s


def create_graph_state(num_qubits):

    """

    :param num_qubits:
    :return:
    """

    hd = qt.hadamard_transform(num_qubits)
    plus_state = 1 / np.sqrt(2) * (qt.basis(2, 0) + qt.basis(2, 1))
    graph_state = qt.tensor([plus_state] * num_qubits)
    for i in range(num_qubits - 1):
        CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=i, target=i + 1, control_value=1)
        graph_state = CZ * graph_state
    CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=num_qubits - 1, target=0, control_value=1)
    graph_state = hd * CZ * graph_state
    return graph_state


def create_box_state(num_qubits):

    """

    :param num_qubits:
    :return:
    """

    hd = qt.hadamard_transform(num_qubits)
    plus_state = 1 / np.sqrt(2) * (qt.basis(2, 0) + qt.basis(2, 1))
    graph_state = qt.tensor([plus_state] * num_qubits)
    assert num_qubits % 2 == 0
    for i in range(3):
        CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=i, target=i + 1, control_value=1)
        graph_state = CZ * graph_state
    CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=num_qubits - 1, target=0, control_value=1)
    graph_state = hd * CZ * graph_state
    CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=2, target=4, control_value=1)
    graph_state = CZ * graph_state
    CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=4, target=5, control_value=1)
    graph_state = CZ * graph_state
    CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=3, target=5, control_value=1)
    graph_state = CZ * graph_state

    return graph_state


def create_ghz_state(num_qubits):

    """

    :param num_qubits:
    :return:
    """

    hd = qt.hadamard_transform(num_qubits)
    plus_state = 1 / np.sqrt(2) * (qt.basis(2, 0) + qt.basis(2, 1))
    graph_state = qt.tensor([plus_state] * num_qubits)
    for i in range(num_qubits):
        for j in range(i):
            CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=i, target=j, control_value=1)
            graph_state = CZ * graph_state
    graph_state = hd * graph_state
    return graph_state


def create_strange_state(num_qubits):

    """

    :param num_qubits:
    :return:
    """

    hd = qt.hadamard_transform(num_qubits)
    plus_state = 1 / np.sqrt(2) * (qt.basis(2, 0) + qt.basis(2, 1))
    cluster_state = qt.tensor([plus_state] * num_qubits)
    for i in range(0, num_qubits - 1):
        print(i)
        CZ = qt.controlled_gate(qt.sigmaz(), N=num_qubits, control=i, target=i + 1, control_value=1)
        cluster_state = CZ * cluster_state
    cluster_state = hd * cluster_state
    print(cluster_state)
    return cluster_state


def tensor(qubit_list):
    s = None

    for i_el, el in enumerate(qubit_list):
        if i_el == 0:
            s = el
        else:
            s = np.kron(s, el)

    return s

def create_w_state(num_qubits):

    """

    :param num_qubits:
    :return:
    """

    qubit_list = [qt.basis(2, 0)] * num_qubits
    qubit_list[0] = qt.basis(2, 1)
    w_state = qt.tensor(qubit_list)
    for i in range(1, num_qubits):
        qubit_list = [qt.basis(2, 0)] * num_qubits
        qubit_list[i] = qt.basis(2, 1)
        w_state += qt.tensor(qubit_list)
    w_state *= 1 / np.sqrt(num_qubits)
    return w_state


def schmidt_rank_index(psi, num_qubits):

    """

    :param psi:
    :param num_qubits:
    :return:
    """

    svr = np.array((0, 0))

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                reduced = (psi.ptrace([i, j])).full()
                rank = np.linalg.matrix_rank(reduced) / 2
                if rank == 1:
                    svr[1] += 1
                if rank == 2:
                    svr[0] += 1

    if num_qubits == 4:
        svr = svr * 1 / 2

    return svr * 1 / 2


def schmidt_rank_vector(psi, num_qubits):

    """

    :param psi:
    :param num_qubits:
    :return:
    """

    svr = np.zeros(num_qubits - 1)

    for i in range(num_qubits - 1):
        reduced = (psi.ptrace([i, i + 1])).full()
        rank = np.linalg.matrix_rank(reduced)
        svr[i] = rank
    return svr
