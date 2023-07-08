import numpy as np
import itertools
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Categorical, Normal
from envs.env_gate_design import IonGatesCircuit
from quantum_circuits.unitary_processes import xxz
from envs.env_utils import construct_cost_function
from ray import tune


class ReinforceAgent(nn.Module):

    def __init__(self, num_inputs: int, num_actions: int, learning_rate=3e-4, beta_softmax=0.01,
                 continuous=False, actor_layers=(64,) * 2, dtype=torch.double, gamma=0.9, max_episodes=20000,
                 device="cpu", seed=0):

        """
        REINFORCE algorithm.
        :param num_inputs: Number of inputs to the network (int).
        :param num_actions: Number of actions to take (int).
        :param learning_rate: Learning rate for the optimizers (float).
        :param beta_softmax: Temperature parameter for the softmax (float).
        :param continuous: Whether the action space is continuous or discrete (bool).
        :param actor_layers: Number of neurons in each layer of the actor network.
        :param dtype: Data type for the network.
        :param gamma: Discount factor.
        :param max_episodes: Maximum number of episodes to run.
        :param device: Device to run the network on.
        :param seed: Seed for the random number generator.
        """

        super(ReinforceAgent, self).__init__()

        self.beta_softmax = beta_softmax
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.continuous = continuous
        actor_dict = OrderedDict()
        self.device = device
        self.dtype = dtype
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.seed = seed
        self.agent_type = "REINFORCE"

        actor_dict["linear_input"] = nn.Linear(self.num_inputs, actor_layers[0])
        actor_dict["relu_input"] = nn.ReLU()
        actor_dict["linear_output"] = nn.Linear(actor_layers[-1], num_actions)

        self.actor = (
            nn.Sequential(actor_dict).to(self.device).to(self.dtype)
        )

        if continuous:
            self.output_layer = nn.Linear(actor_layers[-1], 2 * num_actions).to(self.dtype)
        else:
            self.output_layer = nn.Linear(actor_layers[-1], num_actions).to(self.dtype)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.memory = None

    def save_checkpoint(self, args_dict: dict, env, rewards, infidelities, circuit_length, angle_data, checkpoint_dir):

        """
        Save the checkpoint.
        :param args_dict: Dictionary of arguments.
        :param env: Environment.
        :param rewards: List of rewards.
        :param infidelities: List of infidelities.
        :param circuit_length: List of circuit lengths.
        :param angle_data: List of angle data.
        :param checkpoint_dir: Directory to save the checkpoint.
        :return: None.
        """

        torch.save(dict(
            flag="reinforce_checkpoint" + "_".join([str(key) + "=" + str(value) for key, value in args_dict.items()]),
            actor=self.actor.state_dict(),
            memory=self.memory, args_dict=args_dict, curriculum=env.curriculum, rewards=rewards,
            infidelities=infidelities, circuit_length=circuit_length, angle_data=angle_data),
                   checkpoint_dir)

    def load_checkpoint(self, checkpoint):

        """
        Load the checkpoint.
        :param checkpoint: Checkpoint to load.
        :return: None.
        """

        assert "flag" in checkpoint
        assert "reinforce_checkpoint" in checkpoint["flag"]

        self.actor.load_state_dict(checkpoint['actor'])
        # self.memory = checkpoint['memory']

    def forward(self, x):

        """
        Forward pass through the network.
        :param x: Input to the network.
        :return: Output of the network.
        """

        out = self.actor(x)
        if not self.continuous:
            return F.softmax(self.beta_softmax * out, dim=2)
        else:
            means = out.reshape(-1, self.num_actions)
            sigmas = out[self.num_actions:2 * self.num_actions]
            return means, sigmas

    def get_action(self, state):

        """
        Get the action from the network.
        :param state: State to get the action from.
        :return: Action.
        """
        state = state.clone().detach().to(self.dtype).unsqueeze(0)
        out = self.forward(state)
        action_distr = Categorical(out) if not self.continuous else Normal(out[0], out[1])
        action = action_distr.sample()
        log_prob_a = action_distr.log_prob(action)

        return log_prob_a, action


def update_reinforce(policy_network, rewards, log_probs):

    """
    Update the policy network.
    :param policy_network: Policy network.
    :param rewards: List of rewards.
    :param log_probs: List of log probabilities.
    :return:
    """

    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + policy_network.gamma ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def train_reinforce_agent(env, reinforce_agent, cost, num_episodes, ep_start=0, checkpoint_dir=None, use_tune=False):

    """
    Train the REINFORCE agent.
    :param env: Environment.
    :param reinforce_agent: REINFORCE agent.
    :param cost: Cost function.
    :param num_episodes: Number of episodes to train for.
    :param ep_start: Starting episode.
    :param checkpoint_dir: Directory to save the checkpoint.
    :return:
    """
    if checkpoint_dir:
        print(checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)
        rewards = checkpoint['rewards']
        infidelities = checkpoint['infidelities']
        circuit_length = checkpoint['circuit_length']
        angle_data = checkpoint['angle_data']
        reinforce_agent.load_state_dicts_and_memory(checkpoint)
        env.curriculum = checkpoint['curriculum']
        ep_start = checkpoint['args_dict']['ep_start']
        num_episodes = checkpoint['args_dict']['num_episodes']

        if use_tune:
            with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
                hypers = json.loads(f.read())
                ep_start = hypers["step"] + 1
    else:
        rewards = np.zeros(num_episodes)
        infidelities = np.zeros(num_episodes)
        circuit_length = np.zeros(num_episodes)
        angle_data = np.zeros((num_episodes, 2 * env.max_len_sequence, 1))

    ep_rewards = []
    seq_data = []
    cwd = os.getcwd()

    for i_episode in range(ep_start, num_episodes):
        reinforce_agent.beta_softmax = reinforce_agent.beta_annealing[i_episode]

        rewards = []
        saved_log_probs = []
        state = env.reset()
        state = torch.tensor(state.flatten(), device=reinforce_agent.device, dtype=reinforce_agent.dtype).unsqueeze(0)
        episode_reward = 0

        for t in itertools.count():  # Don't infinite loop while learning
            log_prob, action = reinforce_agent.get_action(state)
            # print(action)
            # print(action.numpy())
            if cost is not None:
                next_state, reward, done, angles, infidelity = env.step(action.numpy()[0, 0], cost)
                # Samples a transition of hybrid RL-optimization
            else:
                next_state, reward, done, angles, infidelity = env.step(action.numpy()[0, 0])
                # Samples a transition of standard RL

            # print(reward)
            saved_log_probs.append(log_prob)

            state = torch.tensor(state.flatten(), device=reinforce_agent.device, dtype=reinforce_agent.dtype).unsqueeze(
                0)
            rewards.append(reward)
            episode_reward += reward
            # print(ep_reward)
            if done:
                ep_rewards.append(episode_reward)
                update_reinforce(reinforce_agent, rewards, saved_log_probs)
                rewards[i_episode] = episode_reward
                infidelities[i_episode] = infidelity
                circuit_length[i_episode] = len(env.gate_sequence)
                seq_data.append("-".join(env.gate_sequence) + "\n")
                # print(angles)
                angle_data[i_episode, :len(angles), 0] = np.array(angles)
                print(f"Agent {reinforce_agent.seed}, REINFORCE episode {i_episode}/{num_episodes}, Reward: {episode_reward}, "
                      f"Length: {len(env.gate_sequence)}")
                break

        if use_tune:
            with tune.checkpoint_dir(step=i_episode) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                with open(path, "w") as f:
                    f.write(json.dumps({"step": i_episode}))
            tune.report(iterations=i_episode, episode_reward=episode_reward)

        if i_episode % 50 == 0:
            args_dict = dict(num_episodes=num_episodes, ep_start=i_episode)
            cp_dir = os.path.join(cwd, f'checkpoint_{reinforce_agent.agent_type}_agent_{reinforce_agent.seed}.pth.tar')
            print(cp_dir)
            reinforce_agent.save_checkpoint(args_dict, env, rewards, infidelities, circuit_length, angle_data, cp_dir)

    return rewards, circuit_length, seq_data, angle_data, infidelities


def target_unitary(num_qubits, params):
    return xxz(num_qubits, params[0], params[1], params[2])


def main():
    torch.cuda.is_available = lambda: False
    n_qubits = 3
    n_episodes = 10
    tg = target_unitary(n_qubits, [1, 0.1, 1])
    gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries\
        = construct_cost_function("MS_singleZ", "numba",
                                                                                      n_qubits, tg, time_dep_u=True)

    env = IonGatesCircuit(target_gate=tg, num_qubits=n_qubits, gate_names=gate_names, max_len_sequence=50, x_opt=x_opt,
                          state_output="circuit", library="numba",
                          threshold=0.03, min_gates=1, n_shots=5)

    reinforce_config = dict(gamma=0.9, learning_rate=0.001, max_episodes=n_episodes, actor_layers=[64, 64])
    agent = ReinforceAgent(env.state_dimension, env.num_actions, **reinforce_config)
    agent.beta_annealing = np.linspace(0.01, 1, n_episodes)

    train_reinforce_agent(env, agent, vec_cost_grad, n_episodes)


if __name__ == "__main__":
    main()
