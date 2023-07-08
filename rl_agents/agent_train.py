import numpy as np
import torch
import copy
import os
import matplotlib.pyplot as plt
import datetime
import json
import time
import argparse

from functools import partial
from rl_agents.new_nn_ps import DeepPSAgent
from rl_agents.ps_agent_flexible import FlexiblePSAgent, train_ps_agent
from rl_agents.lstm_ps import LSTMPSAgent, train_lstm_ps_agent
from rl_agents.ppo import PPO, train_ppo_agent
from rl_agents.reinforce import ReinforceAgent, train_reinforce_agent
from rl_agents.vanilla_pg import VanillaPGAgent, train_vanilla_pg_agent
from joblib import Parallel, delayed
from envs.env_gate_design import IonGatesCircuit, MultiIonGatesCircuit
from envs.env_utils import construct_cost_function
from quantum_circuits.unitary_processes import ucc_operator, target_unitary

import qutip as qt


def convert_state(state: np.ndarray, agent_type: str) -> torch.Tensor:
    """
    Convert the state to the correct format for the agent
    :param state: state of the environment. In this case, it is a list of integers representing the quantum_circuits on the circuit.
    :param agent_type: type of agent. In this case, it is either a DeepPSAgent or a FlexiblePSAgent.
    :return: state in the correct format for the agent.
    """

    if agent_type == "PS-NN":
        state = torch.tensor(state.flatten()).unsqueeze(0)
    elif agent_type == "PS-LSTM":
        state = torch.tensor(state.flatten()).view(1, 1, -1)
        # print(state.shape)
    else:
        pass

    return state


class MultiAgent:
    """ Class for training multiple agents in parallel. """

    def __init__(self, num_agents, env, cost_grad, args, num_cores=5):
        """ Initialize the multi-agent environment.
        :param num_agents: Number of agents to be trained.
        :param env: Environment to be used for training.
        :param cost_grad: Function to compute the gradient of the cost function.
        :param args: Arguments for the training.
        :param num_cores: Number of cores to be used for parallel training.
        """
        self.agents = []
        self.envs = []
        self.num_agents = num_agents
        self.num_cores = num_cores
        self.cost_grad = cost_grad

        for i_agent in range(num_agents):
            agent, _ = create_agent(env, args, seed=args.seed + i_agent)
            self.agents.append(agent)
            self.envs.append(copy.deepcopy(env))

        self.multi_env = MultiIonGatesCircuit(self.envs, cost_grad)

        _, train_func = create_agent(env, args, seed=0)
        self.train_func = train_func

    def set_beta_annealing(self, beta_value):
        """ Set the beta value for the agents.
        beta_value:     Value of beta to be set.
        This is used for the annealing of the beta value in the PS agents."""

        for ag in self.agents:
            ag.beta_softmax = beta_value

    def simulate_sequential(self, n_agents, n_episodes, cost, ep_start=0, checkpoint_dir=None):

        """
        Simulate the training of the agents sequentially.
        :param n_agents: Number of agents to be trained.
        :param n_episodes: Number of episodes to be trained.
        :param cost: Cost function to be used for training.
        :param ep_start: Starting episode number. Used for resuming training.
        :param checkpoint_dir: Directory to save the checkpoints.
        :return: List of costs for each episode.
        """

        assert n_agents <= len(self.agents)
        rew_arr = np.zeros((n_agents, n_episodes))
        infid_arr = np.zeros((n_agents, n_episodes))
        cl_arr = np.zeros((n_agents, n_episodes))
        angle_data_arr = np.zeros((n_agents, n_episodes, 2000, 1))
        seq_data_list = []

        for i_agent, agent in enumerate(self.agents[:n_agents]):
            print(i_agent)
            env = self.envs[i_agent]
            rewards, circuit_length, seq_data, angle_data, infidelities = self.train_func(env, agent, cost, n_episodes,
                                                                                          ep_start,
                                                                                          checkpoint_dir=checkpoint_dir)
            rew_arr[i_agent, :] = rewards
            infid_arr[i_agent, :] = infidelities
            cl_arr[i_agent, :] = circuit_length
            # print(angle_data.shape)
            angle_data_arr[i_agent, :, :angle_data.shape[1], :] = angle_data
            seq_data_list += seq_data

        return self.agents, rew_arr, cl_arr, angle_data_arr, seq_data_list, infid_arr


def create_env(parse_args: argparse.Namespace) -> (MultiIonGatesCircuit, callable):
    """
    Create the environment for the training.
    :param parse_args: Arguments for the training.
    :return: Environment for the training, vectorized cost function
    """

    if parse_args.target_name == "UCC":
        tg = ucc_operator(n_qubits=parse_args.num_qubits, alpha=np.pi / 2)
    elif parse_args.target_name == "XXZ":
        J = parse_args.J
        A = parse_args.A
        Delta = parse_args.Delta
        t = parse_args.evo_time
        print(J, Delta, A, t)
        tg = target_unitary(parse_args.num_qubits, [A, J, Delta], t=t)
        print(np.round(tg, 2))
    else:
        tg = qt.toffoli(parse_args.num_qubits).full()

    if parse_args.library == "numba":
        pass
    else:
        tg = np.array([np.identity(2 ** parse_args.num_qubits), tg])

    gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries = construct_cost_function("standard", parse_args.library,
                                                                                      parse_args.num_qubits, tg,
                                                                                      time_dep_u=True)
    env = IonGatesCircuit(target_gate=tg, num_qubits=parse_args.num_qubits, gate_names=gate_names, x_opt=x_opt,
                          max_len_sequence=parse_args.len_seq,
                          state_output=parse_args.state_output,
                          pop_heuristic=bool(parse_args.pop_heuristic), simplify_state=bool(parse_args.simplify_state),
                          seed=0, library=parse_args.library,
                          threshold=parse_args.threshold, min_gates=parse_args.min_gates, n_shots=parse_args.n_shots,
                          max_iter=parse_args.opt_iterations,
                          curriculum_window=parse_args.curriculum_window, min_threshold=parse_args.minimum_threshold)

    return env, vec_cost_grad

def create_agent(env, parse_args: argparse.Namespace, seed: int, use_tune=False):
    """
    Create the agent for the training.
    :param env: Environment for the training.
    :param parse_args: Arguments for the training.
    :param seed: Seed for the training.
    :param use_tune: Whether to use ray tune for the training.
    :return: Agent for the training, training function for the agent.
    """

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if parse_args.agent_type == "PS-NN":
        parse_args.state_output = "circuit"
        agent = DeepPSAgent(state_dimension=env.state_dimension, num_actions=env.num_actions,
                            beta_softmax=parse_args.beta_softmax, learning_rate=parse_args.learning_rate,
                            hidden_dim=parse_args.hidden_dim,
                            batch_size=parse_args.batch_size, eta_glow_damping=parse_args.eta_glow,
                            num_layers=parse_args.num_layers,
                            gamma_damping=parse_args.gamma_damping,
                            target_update=parse_args.target_update,
                            device="cpu", capacity=parse_args.capacity, replay_time=parse_args.replay_time,
                            constant_nn_dim=bool(parse_args.constant_nn_dim), seed=parse_args.seed)
        train_func = partial(train_lstm_ps_agent, use_tune=use_tune)
    elif parse_args.agent_type == "PS-LSTM":
        parse_args.state_output = "lstm_circuit"
        env.state_output = parse_args.state_output
        agent = LSTMPSAgent(state_dimension=env.state_dimension, num_actions=env.num_actions, output_dim=1,
                            hidden_size=parse_args.hidden_dim,
                            target_update=parse_args.target_update,
                            device="cpu", learning_rate=parse_args.learning_rate, capacity=parse_args.capacity,
                            eta_glow_damping=parse_args.eta_glow, beta_softmax=parse_args.beta_softmax,
                            batch_size=parse_args.batch_size,
                            replay_time=parse_args.replay_time, max_len_sequence=parse_args.len_seq,
                            max_episodes=parse_args.num_episodes, seed=parse_args.seed)
        train_func = partial(train_lstm_ps_agent, use_tune=use_tune)

    elif parse_args.agent_type == "PS":
        parse_args.state_output = "circuit"
        agent = FlexiblePSAgent(env.num_actions, gamma_damping=parse_args.gamma_damping,
                                eta_glow_damping=parse_args.eta_glow, policy_type="softmax",
                                beta_softmax=parse_args.beta_softmax, seed=parse_args.seed)
        train_func = partial(train_ps_agent, use_tune=use_tune)
    elif parse_args.agent_type == "PPO":
        parse_args.state_output = "circuit"
        agent = PPO(state_dim=env.state_dimension, action_dim=env.num_actions, betas=(0.9, 0.999), gamma=0.99,
                    K_epochs=parse_args.len_seq * 4, eps_clip=0.2,
                    actor_layers=(parse_args.hidden_dim,) * parse_args.num_layers,
                    critic_layers=(parse_args.hidden_dim,) * parse_args.num_layers, seed=parse_args.seed)
        train_func = partial(train_ppo_agent, use_tune=use_tune)
    elif parse_args.agent_type == "REINFORCE":
        parse_args.state_output = "circuit"
        agent = ReinforceAgent(num_inputs=env.state_dimension, num_actions=env.num_actions,
                               learning_rate=parse_args.learning_rate, beta_softmax=parse_args.beta_softmax,
                               actor_layers=[parse_args.hidden_dim] * parse_args.num_layers, seed=parse_args.seed)
        train_func = partial(train_reinforce_agent, use_tune=use_tune)
    elif parse_args.agent_type == "VanillaPG":
        parse_args.state_output = "circuit"
        agent = VanillaPGAgent(num_inputs=env.state_dimension, num_actions=env.num_actions,
                               num_hidden=parse_args.hidden_dim,
                               num_layers=parse_args.num_layers, learning_rate=parse_args.learning_rate,
                               beta_softmax=parse_args.beta_softmax, seed=parse_args.seed)
        train_func = partial(train_vanilla_pg_agent, use_tune=use_tune)
    else:
        raise NotImplementedError("This type of RL agent is not implemented")

    agent.beta_annealing = np.linspace(parse_args.beta_softmax, parse_args.beta_max, parse_args.num_episodes)

    return agent, train_func


def set_simulation_folders(parse_args, index):
    cwd = os.getcwd()
    print(f"Simultaion folder {cwd}")

    # Create folder for the reinforcement learning data
    data = os.path.join(cwd, "data_rl")
    if not os.path.exists(data):
        os.mkdir(data)

    # Create folder for the agent based on a given index
    agent_dir = os.path.join(data, f"agent_{index}")
    if not os.path.exists(agent_dir):
        os.mkdir(agent_dir)

    # Save the command line arguments
    with open('backup/commandline_args.txt', 'w') as f:
        json.dump(parse_args.__dict__, f, indent=2)

    return agent_dir


def running_mean(plot: np.ndarray, time_window: int) -> np.ndarray:
    """
    Compute the running mean of a given plot.
    :param plot: np.ndarray of shape (num_episodes, )
    :param time_window: int for the time window
    :return: np.ndarray of shape (num_episodes - 2 * time_window, )
    """

    num_episodes: int = len(plot)
    new_plot = []

    for i in range(time_window, num_episodes - time_window - 1):
        avg_value = np.mean(plot[i - time_window:i])
        new_plot.append(avg_value)

    return new_plot


def train_n_agents_par(args, env, train_func, agent_list):
    # Hyperparameters
    max_len_seq = args.len_seq
    num_qubits = args.num_qubits
    max_episodes = args.num_episodes
    seq_data = []
    beta_annealing = np.linspace(args.beta_softmax, args.beta_max, max_episodes)

    # with threadpool_limits(limits=1, user_api='blas'):
    with Parallel(n_jobs=-2, prefer="processes") as parallel:
        res = parallel([delayed(train_func)(env, agent) for agent in agent_list])

    plotlist = [res[0] for i_agent, _ in enumerate(agent_list)]
    circuit_length = [res[1] for i_agent, _ in enumerate(agent_list)]
    seq_data = [res[2] for i_agent, _ in enumerate(agent_list)]
    angle_data = [res[3] for i_agent, _ in enumerate(agent_list)]

    return agent_list, plotlist, circuit_length, angle_data, seq_data


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def load_checkpoint(model, resume_weights, cuda=False):
    # cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(resume_weights,
                                map_location=lambda storage,
                                                    loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))


def run_multi_agents_seq(env, cost_f, args, cwd, folder=None, checkpoint_folder=None):
    # Simulation data folder
    seed = args.seed
    time_window = 20
    num_agents = args.num_agents
    multi_agent = MultiAgent(num_agents, env, cost_f, args, num_cores=5)
    ep_start = 0
    t0 = time.time()

    agent_list, rew_arr, cl_arr, angle_data_arr, seq_data_list, infid_arr = \
        multi_agent.simulate_sequential(args.num_agents, args.num_episodes, cost_f, ep_start,
                                        checkpoint_dir=checkpoint_folder)
    t1 = time.time() - t0

    if folder:
        data = os.path.join(cwd, folder)
        if not os.path.exists(data):
            os.mkdir(data)
        agent_dir = os.path.join(data, "data_" + "_".join(str(v) for v in args.__dict__.values()))
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
    else:
        data = os.path.join(cwd)
        now = datetime.datetime.now()
        if not os.path.exists(data):
            os.mkdir(data)
        agent_dir = os.path.join(data, "data" + now.strftime("%m_%d_%Y_%H_%M_%S"))
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)

    ep_reward_mean = np.mean(rew_arr, axis=0)
    ep_reward_std = np.std(rew_arr, axis=0) / np.sqrt(num_agents)

    cl_mean = np.mean(cl_arr, axis=0)
    cl_std = np.std(cl_arr, axis=0) / np.sqrt(num_agents)

    fid_mean = np.mean(1 - infid_arr, axis=0)
    fid_std = np.std(1 - infid_arr, axis=0) / np.sqrt(num_agents)

    ep_time_mean = running_mean(ep_reward_mean, time_window)
    cl_time_mean = running_mean(cl_mean, time_window)
    fid_time_mean = running_mean(fid_mean, time_window)

    plt.clf()
    plt.style.use("seaborn-colorblind")
    plt.plot(ep_reward_mean, label="Average reward")
    plt.fill_between(ep_reward_mean, ep_reward_mean - ep_reward_std, ep_reward_mean + ep_reward_std, alpha=0.5)
    plt.plot(ep_time_mean, label="Running average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(agent_dir, "Average reward.png"))

    plt.clf()
    plt.style.use("seaborn-colorblind")
    plt.plot(cl_mean, label="Average length")
    plt.fill_between(cl_mean, cl_mean - cl_std, cl_mean + cl_std, alpha=0.5)
    plt.plot(cl_time_mean, label="Running average")
    plt.xlabel("Episode")
    plt.ylabel("Circuit length")
    plt.legend()
    plt.savefig(os.path.join(agent_dir, "Circuit length.png"))

    plt.clf()
    plt.style.use("seaborn-colorblind")
    plt.plot(fid_mean, label="Average fidelity")
    plt.fill_between(fid_mean, fid_mean - fid_std, fid_mean + fid_std, alpha=0.5)
    plt.plot(fid_time_mean, label="Running average")
    plt.xlabel("Episode")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.savefig(os.path.join(agent_dir, "Fidelity.png"))

    np.savetxt(os.path.join(agent_dir, "fidelities.txt"), 1 - infid_arr)
    np.savetxt(os.path.join(agent_dir, "reward.txt"), rew_arr)
    np.savetxt(os.path.join(agent_dir, "circuit_length.txt"), cl_arr)
    np.savetxt(os.path.join(agent_dir, "angle_params.txt"), angle_data_arr.flatten())

    with open(os.path.join(agent_dir, "sequences.txt"), "w+") as f:
        for i_seq, seq in enumerate(seq_data_list):
            f.write("".join(seq))

    with open(os.path.join(agent_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        f.write(f"\n")
        f.write(f"sim time = {t1}\n")
        f.write(f"state = {env.target_gate}\n")

    fid_list = 1 - infid_arr.flatten()
    print(fid_list)
    print(seq_data_list)
    angle_data = angle_data_arr.reshape(-1, angle_data_arr.shape[2])

    best_fid_idx = np.where(1 - np.asarray(fid_list) < 1e-2)[0]
    print(best_fid_idx)

    if len(best_fid_idx) > 0:
        best_sequences = list(np.asarray(seq_data_list)[best_fid_idx])
        best_fidelities = fid_list[best_fid_idx]
        best_circuit_length = cl_arr.flatten()[best_fid_idx]
        best_params = angle_data[best_fid_idx, :]

        # Best sequences
        np.savetxt(os.path.join(agent_dir, "best_fidelities.txt"), best_fidelities)
        np.savetxt(os.path.join(agent_dir, "best_params.txt"), best_params.flatten())

        with open(os.path.join(agent_dir, "best_sequences.txt"), "w+") as f:
            for i_seq, seq in enumerate(best_sequences):
                f.write("".join(seq))
        best_sequence_idx = np.argmin(best_circuit_length)
        print(best_circuit_length[best_sequence_idx])

        with open(os.path.join(agent_dir, "best_result.txt"), "w+") as f:
            f.write("".join(best_sequences[best_sequence_idx]))
            f.write(str(best_fidelities[best_sequence_idx]))
            f.write(str(best_params[best_sequence_idx, :].flatten()))

        print(f"Shortest correct circuit: {best_sequences[best_sequence_idx]}")
        print(f"with fidelity: {best_fidelities[best_sequence_idx]}")
    else:
        print(f"No sequence found!")

    print("Finished!")
    print(f"Seed: {seed}")
    print(agent_dir)
