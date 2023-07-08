# Code written based upon the implementation found in https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

from ray import tune
from torch.distributions import MultivariateNormal, Categorical
from collections import OrderedDict
from envs.env_gate_design import IonGatesCircuit
from quantum_circuits.unitary_processes import xxz
from envs.env_utils import construct_cost_function
from typing import ClassVar, Tuple


class Memory:

    def __init__(self):

        """
        Initializes the memory.
        """
        # action, state, logarithm of probabilities, reward, and is_terminal (True, False) lists
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):

        """
        Clears the memory lists.
        :return:
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_outputs, actor_layer_sizes, critic_layer_sizes, norm_layers=False,
                 continuous=True, action_std=0.1, device="cpu", dtype=torch.double, beta_softmax=1):

        """
        Initializes the actor-critic model.
        :param num_inputs: The number of inputs to the model.
        :param num_outputs: The number of outputs from the model.
        :param actor_layer_sizes: The sizes of the layers in the actor network.
        :param critic_layer_sizes: The sizes of the layers in the critic network.
        :param norm_layers: Whether to normalize the layers.
        :param continuous: Whether the action space is continuous.
        :param action_std: The standard deviation of the action space.
        :param device: The device to run the model on.
        :param dtype: The data type to use for the model.
        :param beta_softmax: The temperature parameter for the softmax.
        """

        super(ActorCritic, self).__init__()

        self.device = device
        self.dtype = dtype
        self.beta_softmax = beta_softmax

        actor_dict = OrderedDict()
        actor_dict["linear_input"] = nn.Linear(num_inputs, actor_layer_sizes[0])
        if norm_layers:
            print("Norming layers...")
            actor_dict[f"layer_norm"] = nn.LayerNorm(actor_layer_sizes[0])
        actor_dict["relu_input"] = nn.ReLU()

        for i_el, el in enumerate(actor_layer_sizes[1:]):
            actor_dict[f"linear_{i_el}"] = nn.Linear(actor_layer_sizes[i_el - 1], actor_layer_sizes[i_el])
            if norm_layers:
                actor_dict[f"layer_norm_{i_el}"] = nn.LayerNorm(actor_layer_sizes[i_el])
            actor_dict[f"relu_{i_el}"] = nn.ReLU()

        # actor_dict["dropout"] = nn.Dropout(p=0.2)
        actor_dict["linear_output"] = nn.Linear(actor_layer_sizes[1], num_outputs)
        actor_dict["tanh_output"] = nn.Tanh()

        self.actor = (
            nn.Sequential(actor_dict).to(device).to(dtype)
        )
        # critic
        critic_dict = OrderedDict()
        critic_dict["linear_input"] = nn.Linear(num_inputs + num_outputs,
                                                critic_layer_sizes[0])
        # if norm_layers:
        #    print("Norming layers...")
        #    critic_dict["layer_norm"] = nn.LayerNorm(critic_layer_sizes[0])
        critic_dict["relu_input"] = nn.ReLU()

        for i_el, el in enumerate(critic_layer_sizes[1:]):
            critic_dict[f"linear_{i_el}"] = nn.Linear(critic_layer_sizes[i_el - 1], critic_layer_sizes[i_el])
            # if norm_layers:
            #    critic_dict[f"layer_norm_{i_el}"] = nn.LayerNorm(critic_layer_sizes[i_el])
            critic_dict[f"relu_{i_el}"] = nn.ReLU()

        # critic_dict["dropout"] = nn.Dropout(p=0.2)
        critic_dict["linear_output"] = nn.Linear(critic_layer_sizes[1], 1)
        self.critic = (nn.Sequential(critic_dict).to(device).to(dtype))
        self.num_outputs = num_outputs
        self.softmax = nn.Softmax(dim=1)
        if continuous:
            self.action_var = torch.full((num_outputs,), action_std * action_std).to(device)

    def forward(self):

        """
        Forward pass of the model.
        :return:
        """

        raise NotImplementedError

    def act_continuous(self, state: torch.Tensor, memory: Memory) -> torch.Tensor:

        """
        Performs an action in a continuous action space.
        :param state: The state to perform the action in.
        :param memory: The memory to store the action in.
        :return: The action taken (detached tensor).
        """

        action_mean = self.actor(state.to(self.dtype)).to(self.dtype)
        cov_mat = torch.diag(self.action_var).to(self.device).to(self.dtype)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def act_discrete(self, state: torch.Tensor, memory: Memory) -> torch.Tensor:

        """
        Performs an action in a discrete action space.
        :param state: The state of the environment (torch.Tensor).
        :param memory: The memory of the agent (Memory).
        :return: The action taken (torch.Tensor).
        """

        # print(state)
        out = self.actor(state.to(self.dtype)).to(self.dtype)
        # print(out.shape)
        # print(out)
        sm_out = self.softmax(self.beta_softmax * out)
        # print(torch.sum(sm_out))
        # print(sm_out)
        dist = Categorical(sm_out)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate_discrete(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Evaluates the action in a discrete action space.
        :param state: The state of the environment (torch.Tensor).
        :param action: The action to evaluate (torch.Tensor).
        :return: Tuple of the log probability of the action and the entropy of the action.
        """
        out = self.actor(state.to(self.dtype)).to(self.dtype)
        # print(out.shape)
        sm_out = self.softmax(out)
        # print(torch.sum(sm_out))
        dist = Categorical(sm_out)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # print(state.shape)
        action_onehot = F.one_hot(action, num_classes=self.num_outputs).to(self.dtype)
        # print(action_onehot.shape)
        state_value = self.critic(torch.cat([state, action_onehot], dim=1))

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def evaluate_cont(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Evaluates the action in a continuous action space.
        :param state: The state of the environment (torch.Tensor).
        :param action: The action to evaluate (torch.Tensor).
        :return: Tuple of the log probability of the action and the entropy of the action.
        """
        action_mean = self.actor(state.to(self.dtype)).to(self.dtype)

        action_var = self.action_var.expand_as(action_mean).to(self.dtype)
        cov_mat = torch.diag_embed(action_var).to(self.device).to(self.dtype)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # print(state.shape)

        state_value = self.critic(torch.cat([state, action], dim=1))

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, betas=(0.9, 0.999), gamma=0.9, K_epochs=8000, eps_clip=0.2,
                 actor_layers=(64,) * 2, max_episodes=20000, update_freq=200,
                 critic_layers=(64,) * 2, device="cpu", dtype=torch.double, seed=0, beta_softmax=0.001):

        """
        Initializes the PPO class.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param lr: Learning rate.
        :param betas: Beta values for the Adam optimizers.
        :param gamma: Discount factor.
        :param K_epochs: Number of epochs.
        :param eps_clip: Clipping parameter.
        :param actor_layers: Number of layers in the actor network.
        :param max_episodes: Maximum number of episodes.
        :param update_freq: Frequency of updates.
        :param critic_layers: Number of layers in the critic network.
        :param device: Device to use.
        :param dtype: Data type to use.
        :param seed: Seed for the random number generator.
        """

        self.lr_actor = lr_actor
        self.lr_critics = lr_critic
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_freq = update_freq
        self.max_episodes = max_episodes
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.beta_softmax = beta_softmax

        self.policy = ActorCritic(state_dim, action_dim, actor_layers, critic_layers).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = ActorCritic(state_dim, action_dim, actor_layers, critic_layers).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.act = self.policy_old.act_discrete
        self.agent_type = "PPO"

    def select_action(self, state, memory):

        """
        Selects an action.
        :param state: State of the environment.
        :param memory: Memory of the agent.
        :return: Action.
        """
        self.policy.beta_softmax = self.beta_softmax
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.act(state, memory).cpu().data.numpy().flatten()

    def save_checkpoint(self, args_dict: dict, env: ClassVar, rewards: list, infidelities: list, circuit_length: list,
                        angle_data: list, memory: Memory, checkpoint_dir: str):

        """
        Saves the checkpoint.
        :param args_dict: Dictionary of arguments (hyperparameters).
        :param env:  RL environment.
        :param rewards: List of rewards.
        :param infidelities:    List of infidelities.
        :param circuit_length:
        :param angle_data:
        :param memory:
        :param checkpoint_dir:
        :return:
        """

        torch.save(dict(flag="ppo_checkpoint" + "_".join([str(key) + "=" + str(value) for key, value in
                                                          args_dict.items()]),
                        ppo_policy=self.policy.state_dict(), ppo_target_policy=self.policy_old.state_dict(),
                        memory=memory, args_dict=args_dict, curriculum=env.curriculum, rewards=rewards,
                        infidelities=infidelities, circuit_length=circuit_length, angle_data=angle_data),
                   checkpoint_dir)

    def load_checkpoint(self, checkpoint):

        """
        Loads the checkpoint.
        :param checkpoint:
        :return: None.
        """
        assert "flag" in checkpoint
        assert "ppo_checkpoint" in checkpoint["flag"]

        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_old.load_state_dict(checkpoint['policy_old'])

    def update(self, memory: Memory):

        """
        Updates the policy.
        :param memory: Memory of the agent.
        :return: None
        """

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate_discrete(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def train_ppo_agent(env, ppo_agent, cost, num_episodes, ep_start=0, checkpoint_dir=None, use_tune=False):

    """
    Trains the PPO agent.
    :param env: RL environment.
    :param ppo_agent: PPO agent.
    :param cost: Cost function.
    :param num_episodes: Number of episodes.
    :param ep_start: Starting episode.
    :param checkpoint_dir: Directory to save the checkpoints.
    :param use_tune: Whether to use ray tune.
    :return: Tuple of rewards, infidelities, circuit lengths, angle data.
    """

    if checkpoint_dir:
        checkpoint = torch.load(checkpoint_dir)
        rewards = checkpoint['rewards']
        infidelities = checkpoint['infidelities']
        circuit_length = checkpoint['circuit_length']
        angle_data = checkpoint['angle_data']
        ppo_agent.load_state_dicts_and_memory(checkpoint)
        env.curriculum = checkpoint['curriculum']
        ep_start = checkpoint['args_dict']['ep_start']
        num_episodes = checkpoint['args_dict']['num_episodes']
        memory_dict = checkpoint['memory']
        memory = Memory()
        for key, value in memory_dict.items():
            setattr(memory, key, value)

        if use_tune:
            with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
                hypers = json.loads(f.read())
                ep_start = hypers["step"] + 1
    else:
        rewards = np.zeros(num_episodes)
        infidelities = np.zeros(num_episodes)
        circuit_length = np.zeros(num_episodes)
        angle_data = np.zeros((num_episodes, 2 * env.max_len_sequence, 1))
        memory = Memory()

    # logging variables
    time_step = 0
    ppo_agent.train_rewards_list = np.zeros(num_episodes)
    all_pulse_sequences = np.zeros((int(num_episodes * env.max_len_sequence), int(2 + env.num_actions)))
    pulse_sequences = np.zeros((int(env.max_len_sequence), int(2 + env.num_actions)))
    best_pulse_sequence = np.zeros((int(env.max_len_sequence), int(2 + env.num_actions)))
    highest_reward = 0
    cwd = os.getcwd()
    assert len(ppo_agent.beta_annealing) == num_episodes
    seq_data = []
    # training loop
    for i_episode in range(ep_start, num_episodes):
        ppo_agent.beta_softmax = ppo_agent.beta_annealing[i_episode]

        ppo_agent.policy.action_var = 0.99 * ppo_agent.policy.action_var
        episode_reward = 0
        state = env.reset()

        for t in itertools.count():
            time_step += 1
            # Running policy_old:
            action = ppo_agent.select_action(state, memory)
            # print(action)
            if cost is not None:
                next_state, reward, done, angles, infidelity = env.step(action[0], cost)
                # Samples a transition of hybrid RL-optimization
            else:
                next_state, reward, done, angles, infidelity = env.step(action[0])
                # Samples a transition of standard RL            # print(reward)
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % ppo_agent.update_freq == 0:
                ppo_agent.update(memory)
                memory.clear_memory()
                time_step = 0

            episode_reward += reward
            if done:
                rewards[i_episode] = episode_reward
                infidelities[i_episode] = infidelity
                circuit_length[i_episode] = len(env.gate_sequence)
                seq_data.append("-".join(env.gate_sequence) + "\n")
                # print(angles)
                angle_data[i_episode, :len(angles), 0] = np.array(angles)

                print(f"Agent {ppo_agent.seed}, PPO episode {i_episode}/{num_episodes}, Reward: {episode_reward}, "
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
            cp_dir = os.path.join(cwd, f'checkpoint_{ppo_agent.agent_type}_agent_{ppo_agent.seed}.pth.tar')
            ppo_agent.save_checkpoint(args_dict, env, rewards, infidelities, circuit_length, angle_data,
                                      memory.__dict__, cp_dir)

        all_pulse_sequences[i_episode * env.max_len_sequence:(i_episode + 1) * env.max_len_sequence, :] \
            = pulse_sequences

    return rewards, circuit_length, seq_data, angle_data, infidelities


def target_unitary(num_qubits, params):
    return xxz(num_qubits, params[0], params[1], params[2])


def main():
    torch.cuda.is_available = lambda: False
    n_qubits = 3
    n_episodes = 200
    tg = target_unitary(n_qubits, [1, 0.1, 1])
    gate_funcs, gate_names, cost_grad, vec_cost_grad, x_opt, cs_to_unitaries\
        = construct_cost_function("MS_singleZ", "numba",
                                                                                      n_qubits, tg, time_dep_u=True)

    env = IonGatesCircuit(target_gate=tg, num_qubits=n_qubits, gate_names=gate_names, max_len_sequence=50, x_opt=x_opt,
                          state_output="circuit", library="numba",
                          threshold=0.03, min_gates=1, n_shots=5)

    ppo_config = dict(lr=3e-4, betas=(0.9, 0.999), gamma=0.9, K_epochs=80, eps_clip=0.2,
                      actor_layers=(64,) * 2, update_freq=4000, max_episodes=n_episodes,
                      critic_layers=(64,) * 2, device="cpu", dtype=torch.double)

    agent = PPO(state_dim=env.state_dimension, action_dim=env.num_actions, **ppo_config)
    agent.beta_annealing = np.linspace(0.01, 1, n_episodes)

    train_ppo_agent(env, agent, vec_cost_grad, num_episodes=n_episodes)


if __name__ == '__main__':
    main()
