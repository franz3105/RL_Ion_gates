import warnings

import numpy as np
import logging
import os
import datetime
import argparse
import jax
import torch
import qutip as qt
import qutip.qip.algorithms.qft as qft

from envs.env_gate_design import IonGatesCircuit
from jax.lib import xla_bridge
from quantum_circuits.unitary_processes import ucc_operator, randU_Haar, w_states_matrix
from rl_agents.agent_train import run_multi_agents_seq
from envs.env_utils import construct_cost_function
from envs.env_gate_design_layers import IonGatesCircuitLayered

torch.cuda.is_available = lambda: False
jax.config.update('jax_enable_x64', True)

# current working directory
cwd = os.getcwd()
data_folder = cwd + "/data"


def none_or_str(value):
    if value == 'None':
        return None
    return value


# set logger
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='IonGatesCircuitLayered')
parser.add_argument("--agent_type", help="Type of agent", default="PS-LSTM")
parser.add_argument("--seed", help="Seeds the random number generator of all the modules",
                    default=0, type=int)
parser.add_argument("--library", help="Acceleration library to use: either numba or jax",
                    default="numba", type=str)
parser.add_argument("--num_agents", help="The ensemble of rl_agents to be averaged over",
                    default=1, type=int)
parser.add_argument("--platform", help="JAX computing platform",
                    default="cpu", type=str)
parser.add_argument("--num_qubits", help="Number of qubits on the circuit",
                    default=3, type=int)
parser.add_argument("--device", help="Device to use for the simulation", default=0, type=int)
parser.add_argument("--opt_iterations", help="Number of optimization iterations", default=500, type=int)
parser.add_argument("--len_seq", help="Maximal length of the sequence", default=40, type=int)
parser.add_argument("--num_episodes", help="Number of episodes", default=100, type=int)
parser.add_argument("--state_output", help="Type of state", default="circuit", type=str)
parser.add_argument("--beta_softmax", help="Beta parameter of softmax function", default=0.001, type=float)
parser.add_argument("--learning_rate", help="Learning rate", default=0.01, type=float)
parser.add_argument("--hidden_dim", help="Number of hidden neurons", default=128, type=int)
parser.add_argument("--eta_glow", help="Discount factor for PS reward", default=0.01, type=float)
parser.add_argument("--num_layers", help="Number of layers", default=2, type=int)
parser.add_argument("--batch_size", help="Size of training batch", default=64, type=int)
parser.add_argument("--beta_max", help="Maximal value of beta annealing schedule", default=5, type=float)
parser.add_argument("--gamma_damping", help="Damping factor for h-values", default=0.001, type=float)
parser.add_argument("--target_update", help="Update frequency of target network", default=50, type=int)
parser.add_argument("--capacity", help="Memory capacity", default=int(2e4), type=int)
parser.add_argument("--replay_time", help="Frequency of additional training steps", default=100, type=int)
parser.add_argument("--simplify_state", help="If the circuits are simplified automatically or not", default=True,
                    type=bool)
parser.add_argument("--threshold", help="Threshold value to assign reward", default=0.1, type=float)
parser.add_argument("--min_gates", help="Minimum number of quantum_circuits, after which the optimization starts",
                    default=1,
                    type=int)
parser.add_argument("--constant_nn_dim", help="Additional dimension of network input", default=0, type=int)
parser.add_argument("--n_shots", help="Number of parallel optimization attempts", default=10, type=int)
parser.add_argument('--checkpoint_dir', help="whether to use or not the checkpoints", default="False", type=str)
parser.add_argument("--target_name", help="Name of the target unitary", default="Toffoli", type=str)
parser.add_argument("--curriculum_window", help="Number of layers in the circuit", default=500, type=int)
parser.add_argument("--minimum_threshold", help="Number of layers in the circuit", default=1e-2, type=int)


def check_devices(platform):
    assert platform in "gpu", "cpu"
    n_devices = jax.devices(platform)

    return n_devices


def main():
    args = parser.parse_args()

    env_names = ("IonGatesCircuit", "IonGatesCircuitLayered")
    assert args.env in env_names, "Environment name not recognized"

    agent_names = ("PS-LSTM", "PS-NN", "PPO", "REINFORCE", "VanillaPG", "PS",
                   )
    assert args.agent_type in agent_names, "Agent type not supported"

    libraries = ("numba", "jax")
    assert args.library in libraries, "Library not supported"

    platforms = ("gpu", "cpu")
    assert args.platform in platforms, "Platform not supported"

    # target_names = ("randU", "QFT", "Toffoli", "UCC", "XXZ", "W matrix")
    # assert args.target_name in target_names, "Target gate not supported"

    assert args.num_qubits >= 2, "Number of qubits must be at least 2"

    if args.library == "numba":
        assert args.platform == "cpu", "Numba only supports CPU"
    elif args.library == "jax" and args.platform == "cpu":
        warnings.WarningMessage("JAX with cpu tends to be slow here")
    else:
        pass

    if args.agent_type == "PS-LSTM":
        args.state_output = "lstm_circuit"
    else:
        if args.state_output == "lstm_circuit":
            args.state_output = "circuit"

    # Trick for quantum state preparation from zero-state. In case it is needed just define the target state
    # here.
    d = 2 ** args.num_qubits
    target_state = np.zeros(d, np.complex128)
    target_state[0] = 1 / np.sqrt(2)
    target_state[-1] = -1 / np.sqrt(2)
    U_state = np.zeros((d, d), np.complex128)
    if args.target_name == "state":  # We can do state preparation with the same code
        U_state[0, :] = d * target_state
        tg = U_state

    # Available unitaries to be optimized
    elif args.target_name == "Toffoli":
        tg = qt.toffoli(args.num_qubits).full()
    elif args.target_name == "UCC":
        tg = ucc_operator(n_qubits=args.num_qubits, alpha=np.pi / 2)
    elif args.target_name == "W matrix":
        tg = w_states_matrix(args.num_qubits)
        if args.num_qubits == 2:
            tg = (1 / np.sqrt(3)) * np.matrix([[0, 1, 1, 1], [1, 0, -1, 1], [1, 1, 0, -1], [1, -1, 1, 0]],
                                              np.complex128)
    elif args.target_name == "QFT":
        tg = qft.qft(args.num_qubits).full()
    elif args.target_name == "randU":
        tg = randU_Haar(2 ** args.num_qubits)
    else:
        tg = qt.toffoli(args.num_qubits).full()

    if args.library == "numba":
        pass
    else:
        tg = np.array([np.identity(2 ** args.num_qubits, np.complex128), tg])

    data_folder_path = os.path.join(cwd, f"data_rl_{args.target_name}")
    if not os.path.exists(data_folder_path):
        os.mkdir(data_folder_path)

    #if args.env == "IonGatesCircuit":
    #    gate_set_type = "standard"
    #elif args.env == "IonGatesCircuitLayered":
    #    gate_set_type = "layers"
    #else:
    #    raise ValueError("Environment name not recognized")

    gate_set_type = "standard"
    gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries = \
        construct_cost_function(gate_set_type, args.library,
                                args.num_qubits, tg,
                                max_iter=args.opt_iterations,
                                device=args.device)

    env_args = dict(target_gate=tg, num_qubits=args.num_qubits, gate_names=gate_names, x_opt=x_opt,
                    max_len_sequence=args.len_seq,
                    state_output=args.state_output, simplify_state=bool(args.simplify_state),
                    seed=0, library=args.library,
                    threshold=args.threshold, min_gates=args.min_gates, n_shots=args.n_shots,
                    max_iter=args.opt_iterations, curriculum_window=args.curriculum_window,
                    min_threshold=args.minimum_threshold)

    compilation_args = dict(target_gate=tg, num_qubits=args.num_qubits, gate_names=gate_names,
                            x_opt=x_opt,
                            max_len_sequence=args.len_seq,
                            state_output=args.state_output,
                            simplify_state=bool(args.simplify_state),
                            seed=0, library=args.library,
                            threshold=args.threshold, min_gates=args.min_gates, n_shots=args.n_shots,
                            max_iter=args.opt_iterations)

    if args.env == "IonGatesCircuit":
        env = IonGatesCircuit(**env_args)
    elif args.env == "IonGatesCircuitLayered":
        env = IonGatesCircuitLayered(**env_args)
    else:
        raise NotImplementedError("This environment is not available!")

    # print(args.num_episodes)
    if args.checkpoint_dir == "True":
        checkpoint_dir = os.path.join(cwd, f'checkpoint_{args.agent_type}_agent_{args.seed}.pth.tar')
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = None
    else:
        checkpoint_dir = None

    run_multi_agents_seq(env, vec_cost_grad, args, data_folder_path, checkpoint_folder=checkpoint_dir)

    return


if __name__ == "__main__":
    jax.config.update('jax_platform_name', parser.parse_args().platform)
    local = datetime.datetime.now()
    logging.info(local)
    print(xla_bridge.get_backend().platform)
    main()
