import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import itertools
import os
import json

from collections import OrderedDict
from torch.distributions import Categorical
from envs.env_gate_design import IonGatesCircuit
from quantum_circuits.unitary_processes import xxz
from envs.env_utils import construct_cost_function
from ray import tune

class VanillaPGAgent(nn.Module):

    def __init__(self, num_inputs, num_actions, num_a_continuous=0, num_hidden=100, num_layers=2, learning_rate=1e-2,
                 beta_softmax=1, dtype=torch.double, device="cpu", gamma=0.99, max_episodes=20000, seed=0,
                 actor_layers=(64,) * 2,
                 critic_layers=(64,) * 2):

        """
        Vanilla Policy Gradient agent.
        :param num_inputs: Number of inputs to the network.
        :param num_actions: Number of discrete actions.
        :param num_a_continuous: Number of continuous actions.
        :param num_hidden: Number of hidden units in the network.
        :param num_layers: Number of hidden layers in the network.
        :param learning_rate: Learning rate for the optimizers.
        :param beta_softmax: Softmax temperature for the discrete actions.
        :param dtype: Data type for the network.
        :param device: Device for the network.
        :param gamma: Discount factor for the rewards.
        :param max_episodes: Maximum number of episodes to run.
        :param seed: Seed for the random number generator.
        :param actor_layers: Number of hidden units in the actor network.
        :param critic_layers: Number of hidden units in the critic network.
        """
        super(VanillaPGAgent, self).__init__()

        # Policy (actor) network
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.seed = seed
        self.agent_type = "VanillaPG"

        actor_dict = OrderedDict()
        actor_dict["input"] = nn.Linear(num_inputs, num_hidden)
        actor_dict["input_relu"] = nn.ReLU()
        # nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        for i in range(num_layers):
            actor_dict[f"hidden_{i}"] = nn.Linear(num_hidden, num_hidden)
            actor_dict[f"relu_{i}"] = nn.ReLU()
        # nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")
        actor_dict[f"output"] = nn.Linear(num_hidden, num_actions + num_a_continuous * (num_a_continuous + 1))

        self.actor = nn.Sequential(actor_dict).to(dtype)
        # nn.init.xavier_normal(self.layer2.weight)

        # Critic (value) network
        critic_dict = OrderedDict()
        critic_dict["input"] = nn.Linear(num_inputs, num_hidden)
        critic_dict["input_relu"] = nn.ReLU().to(dtype)
        # nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        for i in range(num_layers):
            critic_dict[f"hidden_{i}"] = nn.Linear(num_hidden, num_hidden)
            critic_dict[f"relu_{i}"] = nn.ReLU()

        critic_dict[f"output"] = nn.Linear(num_hidden, 1)
        # nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")

        self.critic = nn.Sequential(critic_dict).to(dtype)

        # ----------------------------------------------------------------

        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor = copy.deepcopy(self.actor)

        # ------------------------------------------------------------------

        self.relu = nn.ReLU()
        self.beta_softmax = beta_softmax
        self.num_actions = num_actions
        self.saved_log_probs = []
        self.entropies = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.memory = dict(entropies=self.entropies, rewards=self.rewards, values=self.values, returns=self.returns)

    def forward(self, x):

        """
        Forward pass through the network.
        :param x: Input to the network.
        :return:
        """
        out = self.actor(x)
        # print(out)
        # print(out)
        discrete_out = out[0, :, :self.num_actions]

        actor_output = F.softmax(self.beta_softmax * discrete_out, dim=1)
        return actor_output

    def save_checkpoint(self, args_dict, env, rewards, infidelities, circuit_length, angle_data, checkpoint_dir):

        """
        Save the agent's state.
        :param args_dict: Dictionary of arguments.
        :param env: Environment.
        :param rewards: List of rewards.
        :param infidelities: List of infidelities.
        :param circuit_length: List of circuit lengths.
        :param angle_data: List of angle data.
        :param checkpoint_dir: Directory to save the checkpoint.
        :return:
        """
        torch.save(dict(flag="vanillaPG_checkpoint" + "_".join([str(key) + "=" + str(value) for key, value in args_dict.items()]),
                        actor=self.actor.state_dict(), cricic=self.critic.state_dict(),
                        target_actor=self.target_actor.state_dict(), target_critic=self.target_critic.state_dict(),
                        memory=self.memory, args_dict=args_dict, curriculum=env.curriculum, rewards=rewards,
                        infidelities=infidelities, circuit_length=circuit_length, angle_data=angle_data),
                   checkpoint_dir)

    def load_checkpoint(self, checkpoint):

        """
        Load the agent's state.
        :param checkpoint:  Checkpoint to load.
        :return:
        """
        assert "flag" in checkpoint
        assert "vanillaPG_checkpoint" in checkpoint["flag"]

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.memory = checkpoint['memory']

    def get_action(self, state, save_values=True, requires_grad=True):

        """
        Get an action from the agent.
        :param state: State of the environment.
        :param save_values: Whether to save the values.
        :param requires_grad: Whether to require gradients.
        :return: Action.
        """
        state = state.clone().detach().unsqueeze(0)
        out = self.forward(state)
        # print(out)
        discr_action_distr = Categorical(out[0])
        discr_action = discr_action_distr.sample()

        if save_values:
            self.saved_log_probs.append(
                discr_action_distr.log_prob(discr_action))
            self.entropies.append(discr_action_distr.entropy().mean())
            self.values.append(self.critic(state))

        return discr_action  # , cont_action


def update_vanilla_pg(agent, next_state):

    """
    Update the agent.
    :param agent: Agent.
    :param next_state: Next state.
    :return:
    """

    R = agent.critic(next_state)
    policy_loss = []
    for r in agent.rewards[::-1]:
        R = r + agent.gamma * R
        agent.returns.insert(0, R)
    tens_returns = torch.tensor(agent.returns).detach()
    tens_returns = (tens_returns - tens_returns.mean()) / (tens_returns.std() + 1e-9)
    # print(agent.values)
    tens_values = torch.stack(agent.values)

    tens_log_probs = torch.stack(agent.saved_log_probs)
    torch.autograd.set_detect_anomaly(True)
    advantage = tens_values - tens_returns
    actor_loss = - (tens_log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    agent.critic_optimizer.zero_grad()
    agent.actor_optimizer.zero_grad()
    critic_loss.backward()
    actor_loss.backward()
    agent.critic_optimizer.step()
    agent.actor_optimizer.step()

    del agent.rewards[:]
    del agent.saved_log_probs[:]
    del agent.values[:]
    del agent.returns[:]
    del agent.entropies[:]

    return 0


def train_vanilla_pg_agent(env, vanilla_pg_agent: VanillaPGAgent, cost, num_episodes, ep_start=0, checkpoint_dir=None,
                           use_tune=False):

    """
    Train the agent.
    :param env: Environment.
    :param vanilla_pg_agent: Agent.
    :param cost: Cost function.
    :param num_episodes: Number of episodes.
    :param ep_start: Starting episode.
    :param checkpoint_dir: Directory to save the checkpoint.
    :return: Rewards, infidelities, circuit lengths, angle data.
    """

    if checkpoint_dir:
        print(checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)
        rewards = checkpoint['rewards']
        infidelities = checkpoint['infidelities']
        circuit_length = checkpoint['circuit_length']
        angle_data = checkpoint['angle_data']
        vanilla_pg_agent.load_state_dicts_and_memory(checkpoint)
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

    seq_data = []
    cwd = os.getcwd()

    for i_episode in range(ep_start, num_episodes):
        vanilla_pg_agent.beta_softmax = vanilla_pg_agent.beta_annealing[i_episode]

        state = env.reset()
        state = torch.tensor(state.flatten()).unsqueeze(0)
        episode_reward = 0

        for t in itertools.count():
            action = vanilla_pg_agent.get_action(state)
            if cost is not None:
                next_state, reward, done, angles, infidelity = env.step(action.item(), cost)
                # Samples a transition of hybrid RL-optimization
            else:
                next_state, reward, done, angles, infidelity = env.step(action.item())
                # Samples a transition of standard RL
                # next_state = torch.tensor(next_state.flatten()).unsqueeze(0)

            episode_reward += reward

            vanilla_pg_agent.rewards.append(reward)
            next_state = torch.tensor(next_state.flatten()).unsqueeze(0)
            state = next_state

            if done:
                update_vanilla_pg(vanilla_pg_agent, next_state)
                rewards[i_episode] = episode_reward
                infidelities[i_episode] = infidelity
                circuit_length[i_episode] = len(env.gate_sequence)
                seq_data.append("-".join(env.gate_sequence) + "\n")
                # print(angles)
                angle_data[i_episode, :len(angles), 0] = np.array(angles)
                print(f"Agent {vanilla_pg_agent.seed}, Vanilla PG episode {i_episode}/{num_episodes}, Reward: {episode_reward}, "
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
            cp_dir = os.path.join(cwd,
                                  f'checkpoint_{vanilla_pg_agent.agent_type}_agent_{vanilla_pg_agent.seed}.pth.tar')
            print(cp_dir)
            vanilla_pg_agent.save_checkpoint(args_dict, env, rewards, infidelities, circuit_length, angle_data, cp_dir)

    return rewards, circuit_length, seq_data, angle_data, infidelities


def target_unitary(num_qubits, params):
    return xxz(num_qubits, params[0], params[1], params[2])


def main():
    torch.cuda.is_available = lambda: False
    n_qubits = 3
    n_episodes = 20
    tg = target_unitary(n_qubits, [1, 0.1, 1])
    gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries\
        = construct_cost_function("MS_singleZ", "numba",
                                                                                      n_qubits, tg, time_dep_u=True)

    env = IonGatesCircuit(target_gate=tg, num_qubits=n_qubits, gate_names=gate_names, max_len_sequence=50, x_opt=x_opt,
                          state_output="circuit", library="numba",
                          threshold=0.03, min_gates=1, n_shots=5)

    vanilla_pg_config = dict(gamma=0.99, learning_rate=0.001, max_episodes=n_episodes, actor_layers=[64, 64],
                             critic_layers=[64, 64])

    agent = VanillaPGAgent(env.state_dimension, env.num_actions, **vanilla_pg_config)
    agent.beta_annealing = np.linspace(0.01, 1, n_episodes)

    train_vanilla_pg_agent(env, agent, vec_cost_grad, n_episodes)


if __name__ == "__main__":
    main()
