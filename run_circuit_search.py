import argparse
import jax
import torch
import os
import warnings
import qutip as qt
import qutip.qip.algorithms.qft as qft

import numpy as np
from quantum_circuits.unitary_processes import ucc_operator, randU_Haar, w_states_matrix
from circuit_search.compilation_in_layers import LayerCompilation
from envs.env_utils import construct_cost_function
from circuit_search.exhaustive_search import ExhaustiveSearch
from circuit_search.random_search import RandomSearch

torch.cuda.is_available = lambda: False
jax.config.update('jax_enable_x64', True)

# current working directory
cwd = os.getcwd()
data_folder = cwd + "/data"


def none_or_str(value):
    if value == 'None':
        return None
    return value


# Circuit search file

# set logger
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='IonGatesCircuit')
parser.add_argument("--algo_type", help="Type of agent", default="LayerCompilation", type=str)
parser.add_argument("--seed", help="Seeds the random number generator of all the modules",
                    default=0, type=int)
parser.add_argument("--library", help="Acceleration library to use: either numba or jax",
                    default="numba", type=str)
parser.add_argument("--opt_iterations", help="Number of optimization iterations", default=500, type=int)
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--num_agents", help="The ensemble of rl_agents to be averaged over",
                    default=1, type=int)
parser.add_argument("--platform", help="JAX computing platform",
                    default="cpu", type=str)
parser.add_argument("--num_qubits", help="Number of qubits on the circuit",
                    default=3, type=int)
parser.add_argument("--num_ms_gates", help="Maximum number of MS gates to use",
                    default=3, type=int)
parser.add_argument("--device", help="Device to use for the simulation", default=0, type=int)
parser.add_argument("--max_iter", help="Number of optimization iterations", default=100, type=int)
parser.add_argument("--len_seq", help="Maximal length of the sequence", default=50, type=int)
parser.add_argument("--num_episodes", help="Number of episodes", default=1000, type=int)
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
parser.add_argument("--pop_heuristic", help="If the population of the optimization run is random or not",
                    default=False, type=bool)
parser.add_argument("--simplify_state", help="If the circuits are simplified automatically or not", default=True,
                    type=bool)
parser.add_argument("--threshold", help="Threshold value to assign reward", default=0.1, type=float)
parser.add_argument("--min_gates", help="Minimum number of quantum_circuits, after which the optimization starts",
                    default=1, type=int)
parser.add_argument("--constant_nn_dim", help="Additional dimension of network input", default=0, type=int)
parser.add_argument("--n_shots", help="Number of parallel optimization attempts", default=100, type=int)
parser.add_argument('--checkpoint_dir', help="whether to use or not the checkpoints", default="False", type=str)
parser.add_argument("--target_name", help="Name of the target unitary", default="W matrix", type=str)
parser.add_argument("--curriculum_window", help="Number of layers in the circuit", default=500, type=int)
parser.add_argument("--minimum_threshold", help="Number of layers in the circuit", default=1e-2, type=int)


def main():
    args = parser.parse_args()

    assert args.algo_type in ("ExhaustiveSearch", "RandomSearch", "LayerCompilation")
    assert args.num_qubits >= 2, "Number of qubits must be at least 2"

    if args.library == "numba":
        assert args.platform == "cpu", "Numba only supports CPU"
    elif args.library == "jax" and args.platform == "cpu":
        warnings.WarningMessage("JAX with cpu tends to be slow here")
    else:
        pass

    if args.target_name == "UCC":
        tg = ucc_operator(n_qubits=args.num_qubits, alpha=np.pi / 2)
    elif args.target_name == "W matrix":
        tg = w_states_matrix(args.num_qubits)
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
    gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries = construct_cost_function("standard",
                                                                                                       args.library,
                                                                                                       args.num_qubits,
                                                                                                       tg,
                                                                                                       max_iter=args.opt_iterations,
                                                                                                       device=args.device)
    data_folder_path = os.path.join(cwd, f"data_circsearch_{args.target_name}")

    if args.algo_type == "LayerCompilation":
        for i in range(args.num_episodes):
            alg = LayerCompilation(target_gate=tg, num_qubits=args.num_qubits, gate_names=gate_names,
                                   x_opt=x_opt,
                                   max_len_sequence=args.len_seq,
                                   state_output=args.state_output,
                                   pop_heuristic=bool(args.pop_heuristic),
                                   simplify_state=bool(args.simplify_state),
                                   seed=i, library=args.library,
                                   threshold=args.threshold, min_gates=args.min_gates, n_shots=args.n_shots,
                                   max_iter=args.opt_iterations)
            alg.run_compilation()
            alg.save_results(data_folder_path)
        # lp = LineProfiler()
        # lp_wrapper = lp(alg.run_compilation)
        # lp.add_function(alg.optimize_step)
        # lp.add_function(alg.minimize_cost_)
        # lp_wrapper()
        # lp.print_stats()
        # alg.run_compilation()
        # alg.save_results(data_folder_path)

    elif args.algo_type == "ExhaustiveSearch":
        exh_search = ExhaustiveSearch(num_layers=args.num_ms_gates,
                                      target_gate=tg, num_qubits=args.num_qubits, gate_names=gate_names,
                                      x_opt=x_opt,
                                      max_len_sequence=args.len_seq,
                                      state_output=args.state_output,
                                      pop_heuristic=bool(args.pop_heuristic),
                                      simplify_state=bool(args.simplify_state),
                                      seed=0, library=args.library,
                                      threshold=args.threshold, min_gates=args.min_gates, n_shots=args.n_shots,
                                      max_iter=args.opt_iterations)
        exh_search.run_search()
        exh_search.save_results(data_folder_path)

    elif args.algo_type == "RandomSearch":
        rand_search = RandomSearch(target_gate=tg, num_qubits=args.num_qubits, gate_names=gate_names,
                                   x_opt=x_opt, num_episodes=args.num_episodes,
                                   max_len_sequence=args.len_seq, num_layers=args.num_ms_gates,
                                   state_output=args.state_output,
                                   pop_heuristic=bool(args.pop_heuristic),
                                   simplify_state=bool(args.simplify_state),
                                   seed=0, library=args.library,
                                   threshold=args.threshold, min_gates=args.min_gates, n_shots=args.n_shots,
                                   max_iter=args.opt_iterations)
        rand_search.run_search()
        rand_search.save_results(data_folder_path)

    return


if __name__ == '__main__':
    main()
