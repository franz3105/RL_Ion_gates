import logging
import os
import argparse
import jax
import torch
import datetime
import numpy as np

from jax.lib import xla_bridge
from rl_agents.agent_train import run_multi_agents_seq, create_env

# current working directory
cwd = os.getcwd()

# Specific simulation file for the XXZ model unitary evolution.

parser = argparse.ArgumentParser()
parser.add_argument("--J", help="coupling", default=0.1, type=float)
parser.add_argument("--A", help="Frequency", default=0.1, type=float)
parser.add_argument("--Delta", help="inhomogeneity", default=0.5, type=float)
parser.add_argument("--evo_time", help="Evolution time", default=0.25, type=float)
parser.add_argument('--env', type=str, default='IonGatesCircuit')
parser.add_argument("--agent_type", help="Type of agent", default="PS-LSTM")
parser.add_argument("--seed", help="Seeds the random number generator of all the modules",
                    default=0, type=int)
parser.add_argument("--library", help="Acceleration library to use: either numba or jax",
                    default="numba", type=str)
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--num_agents", help="The ensemble of rl_agents to be averaged over",
                    default=3, type=int)
parser.add_argument("--platform", help="JAX computing platform",
                    default="gpu", type=str)
parser.add_argument("--num_qubits", help="Number of qubits on the circuit",
                    default=5, type=int)
parser.add_argument("--device", help="Device to use for the simulation", default=0, type=int)
parser.add_argument("--opt_iterations", help="Number of optimization iterations", default=100, type=int)
parser.add_argument("--len_seq", help="Maximal length of the sequence", default=50, type=int)
parser.add_argument("--num_episodes", help="Number of episodes", default=500, type=int)
parser.add_argument("--state_output", help="Type of state", default="lstm_circuit", type=str)
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
parser.add_argument("--threshold", help="Threshold value to assign reward", default=0.01, type=float)
parser.add_argument("--min_gates", help="Minimum number of quantum_circuits, after which the optimization starts",
                    default=1,
                    type=int)
parser.add_argument("--constant_nn_dim", help="Additional dimension of network input", default=0, type=int)
parser.add_argument("--n_shots", help="Number of parallel optimization attempts", default=5, type=int)
parser.add_argument('--checkpoint_dir', help="whether to use or not the checkpoints", default="False", type=str)
parser.add_argument("--target_name", help="Name of the target unitary", default="XXZ")
parser.add_argument("--curriculum_window", help="Number of layers in the circuit", default=500, type=int)
parser.add_argument("--minimum_threshold", help="Number of layers in the circuit", default=1e-2, type=int)

jax.config.update('jax_enable_x64', True)


def simulate(args):
    env, vec_cost_grad = create_env(args)

    data_folder_path = os.path.join(cwd, f"data_rl_{args.target_name}")

    if not os.path.exists(data_folder_path):
        os.mkdir(data_folder_path)

    qubit_folder_path = os.path.join(data_folder_path, f"data_nq={args.num_qubits}")

    if not os.path.exists(qubit_folder_path):
        os.mkdir(qubit_folder_path)

    params_folder_path = os.path.join(qubit_folder_path, f"data_J={args.J}_A={args.A}_Delta={args.Delta}")

    if not os.path.exists(params_folder_path):
        os.mkdir(params_folder_path)

    # alg.env.set_target_gate(tg)
    # alg.run_compilation()
    # alg.save_results(data_folder_path)

    if args.checkpoint_dir == "True":
        checkpoint_dir = os.path.join(cwd, f'checkpoint_{args.agent_type}_agent_{args.seed}.pth.tar')
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = None
    else:
        checkpoint_dir = None

    run_multi_agents_seq(env, vec_cost_grad, args, cwd, folder=params_folder_path, checkpoint_folder=checkpoint_dir)
    return


if __name__ == "__main__":
    jax.config.update('jax_platform_name', parser.parse_args().platform)
    local = datetime.datetime.now()
    logging.info(local)
    print(xla_bridge.get_backend().platform)
    simulate(parser.parse_args())
