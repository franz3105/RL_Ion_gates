import torch
import os
import datetime
import argparse

from ray import tune
from rl_agents.agent_train import create_agent, create_env

# This file can be used for hyperparameter tuning of the agent with respect to a task

# current working directory
cwd = os.getcwd()
data_folder = cwd + "/data"

# Parser environment initialization
parser = argparse.ArgumentParser()
parser.add_argument("--agent_type", help="Type of agent", default="PS-NN")
parser.add_argument("--seed", help="Seeds the random number generator of all the modules",
                    default=123, type=int)
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--num_agents", help="The ensemble of rl_agents to be averaged over",
                    default=1, type=int)
parser.add_argument("--len_seq", help="Maximal length of the sequence", default=30, type=int)
parser.add_argument("--num_qubits", help="Number of qubits", default=3, type=int)
parser.add_argument("--num_episodes", help="Number of episodes", default=5000, type=int)
parser.add_argument("--state_output", help="Type of state", default="circuit")
parser.add_argument("--capacity", help="Memory capacity", default=int(2e6))
parser.add_argument("--target_name", help="Checkpoint folder", default="UCC")
parser.add_argument("--library", help="Acceleration library to use: either numba or jax",
                    default="numba", type=str)
parser.add_argument("--pop_heuristic", help="If the population of the optimization run is random or not",
                    default=False, type=bool)
parser.add_argument("--simplify_state", help="If the circuits are simplified automatically or not", default=True,
                    type=bool)
parser.add_argument("--threshold", help="Threshold value to assign reward", default=0.03, type=float)
parser.add_argument("--min_gates", help="Minimum number of quantum_circuits, after which the optimization starts", default=1,
                    type=int)
parser.add_argument("--n_shots", help="Number of parallel optimization attempts", default=5, type=int)

args = parser.parse_args()


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    args = parser.parse_args()


def agent_train_cg(env, cost_grad, config, checkpoint_dir=None, seed=0):
    parse_args = Namespace(**config)
    print(parse_args)

    agent, train_func = create_agent(env, parse_args, seed=seed, use_tune=True)

    rewards, circuit_length, seq_data, angle_data, infidelities = train_func(env, agent, cost_grad,
                                                                             config["num_episodes"],
                                                                             config["ep_start"],
                                                                             checkpoint_dir=checkpoint_dir)


def run_hyperparameter_search(agent_name):
    env, cost_grad = create_env(args)
    seed = 0

    def agent_train(config, checkpoint_dir=None):
        return agent_train_cg(config=config, env=env, cost_grad=cost_grad, checkpoint_dir=checkpoint_dir, seed=seed)

    asha_scheduler = tune.schedulers.ASHAScheduler(
        metric='episode_reward',
        mode='max',
        max_t=1000,
        grace_period=10,
        reduction_factor=3,
        brackets=1)

    initial_best_config = {"len_seq": 30,
                           "max_episodes": 20000,
                           "lr": 1e-3,
                           "hidden dim": 128,
                           "batch_size": 32,
                           "beta_softmax": 0.01,
                           "eta_glow": 0.01,
                           "gamma_damping": 0.,
                           "num_layers": 2,
                           "target_update": 50,
                           "replay time": 100,
                           "beta_max": 1}

    # bayesopt = BayesOptSearch(metric="episode_reward", mode="max")
    if agent_name == "PS-LSTM":
        config = {
            "agent_type": agent_name,
            "len_seq": 30,
            "num_episodes": 20000,
            "learning_rate": tune.uniform(1e-3, 1e-1),
            "hidden_dim": tune.grid_search([32, 64, 128]),
            "batch_size": 32,
            "beta_softmax": 0.001,
            "eta_glow": tune.grid_search([0.01, 0.1]),
            "gamma_damping": 0.,
            "num_layers": tune.grid_search([2, 3, 4, 5]),
            "target_update": 100,
            "replay_time": 50,
            "beta_max": tune.grid_search([1, 2, 5]),
            "capacity": int(2e4),
            "seed": seed,
            "ep_start": 0

        }
    elif agent_name == "PS-NN":
        config = {
            "agent_type": agent_name,
            "len_seq": 30,
            "num_episodes": 20000,
            "learning_rate": tune.uniform(1e-3, 1e-1),
            "hidden_dim": tune.grid_search([32, 64, 128]),
            "batch_size": 32,
            "beta_softmax": 0.001,
            "eta_glow": tune.grid_search([0.01, 0.1]),
            "gamma_damping": 0.,
            "num_layers": tune.grid_search([2, 3, 4, 5]),
            "target_update": 100,
            "replay_time": 50,
            "beta_max": tune.grid_search([1, 2, 5]),
            "capacity": int(2e4),
            "seed": seed,
            "ep_start": 0

        }
    elif agent_name == "PPO":
        config = {
            "agent_type": agent_name,
            "len_seq": 30,
            "num_episodes": 20000,
            "learning_rate": tune.uniform(1e-3, 1e-1),
            "hidden_dim": tune.grid_search([32, 64, 128]),
            "batch_size": 32,
            "beta_softmax": 0.001,
            "eta_glow": tune.grid_search([0.01, 0.1]),
            "gamma_damping": 0.,
            "num_layers": tune.grid_search([2, 3, 4, 5]),
            "target_update": 100,
            "replay_time": 50,
            "beta_max": tune.grid_search([1, 2, 5]),
            "capacity": int(2e4),
            "seed": seed,
            "ep_start": 0

        }
    elif agent_name == "REINFORCE":
        config = {
            "agent_type": agent_name,
            "len_seq": 30,
            "num_episodes": 20000,
            "learning_rate": tune.uniform(1e-3, 1e-1),
            "hidden_dim": tune.grid_search([32, 64, 128]),
            "batch_size": 32,
            "beta_softmax": 0.001,
            "eta_glow": tune.grid_search([0.01, 0.1]),
            "gamma_damping": 0.,
            "num_layers": tune.grid_search([2, 3, 4, 5]),
            "target_update": 100,
            "replay_time": 50,
            "beta_max": tune.grid_search([1, 2, 5]),
            "capacity": int(2e4),
            "seed": seed,
            "ep_start": 0

        }
    elif agent_name == "Vanilla PG":
        config = {
            "agent_type": agent_name,
            "len_seq": 30,
            "num_episodes": 20000,
            "learning_rate": tune.uniform(1e-3, 1e-1),
            "hidden_dim": tune.grid_search([32, 64, 128]),
            "batch_size": 32,
            "beta_softmax": 0.001,
            "eta_glow": tune.grid_search([0.01, 0.1]),
            "gamma_damping": 0.,
            "num_layers": tune.grid_search([2, 3, 4, 5]),
            "target_update": 100,
            "replay_time": 50,
            "beta_max": tune.grid_search([1, 2, 5]),
            "capacity": int(2e4),
            "seed": seed,
            "ep_start": 0

        }
    else:
        raise NotImplementedError

    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    print(f"Num of CPUs: {num_cpus}, Number of GPUs: {num_gpus}")
    analysis = tune.run(agent_train, config=config, scheduler=asha_scheduler,
                        resources_per_trial={"cpu": num_cpus, "gpu": num_gpus},
                        local_dir="home/francesco/PhD/Ion_gates/ray_results", )

    print("Best config: ", analysis.get_best_config(
        metric="episode_reward", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    now = datetime.datetime.now()
    df.to_csv("ion_gates_analysis_results" + now.strftime("%m_%d_%Y, %H_%M_%S") + ".csv")

    print(f"Hyperparameter search for agent {agent_name} finished!")


def main():
    agent_name = "PS-LSTM"
    run_hyperparameter_search(agent_name)


if __name__ == "__main__":
    main()
