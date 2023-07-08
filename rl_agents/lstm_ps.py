import os.path
import random
import numpy as np
from collections import deque
from collections import namedtuple
from ray import tune

import json
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('percept', 'action',
                                       'reward'))  # we use namedtuple
# as a data structure to store events with their properties (percept, action, reward)

dtype = torch.double


def gradClamp(model: torch.nn.Module, clip=5):  # in case we need to clamp gradients

    """
    Clamps the gradients.
    :param model: PyTorch model.
    :param clip: value to clip the gradients to.
    :return: clipped gradients.
    """

    nn.utils.clip_grad_norm(model.parameters(), clip)


class NN_LSTM(nn.Module):

    """
    Python super function:
    Use Case 1: Super can be called upon in a single inheritance,
    in order to refer to the parent class or multiple classes without explicitly naming them.
    It’s somewhat of a shortcut, but more importantly, it helps keep your code maintainable for the foreseeable future.

    Use Case 2: Super can be called upon in a dynamic execution environment for multiple or collaborative inheritance.
    This use is considered exclusive to Python, because it’s not possible with languages that only support
    single inheritance or are statically compiled.

    input_size: dimension of the tensor, which is given as input to the neural network

    hidden_size: size of the hidden layers

    output_size: dimension of the output, in case of PS it should be equal to 1, since the output is the real value h(s,a)

    num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1

    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, batch_size=64):
        super(NN_LSTM, self).__init__()
        self.input_size = input_size  # this representation allows better predictive power
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # print(input_size)
        self.num_layers = num_layers
        self.hidden = (torch.zeros((num_layers, 1, hidden_size)).to(dtype),) * 2
        self.batch_hidden = (torch.zeros((self.num_layers, batch_size, hidden_size)).to(dtype),) * 2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # output layer mapping hidden weights to h(s,a): All weights are 1, biases are 0
        self.output = nn.Linear(hidden_size, output_size)
        nn.init.constant_(self.output.weight, 1.0 / hidden_size)
        nn.init.constant_(self.output.bias, 0.)

    def reset_h(self):
        self.hidden = (torch.zeros((self.num_layers, 1, self.hidden_size)).to(dtype),) * 2

    def reset_batch_h(self):
        self.batch_hidden = (torch.zeros((self.num_layers, self.batch_size, self.hidden_size)).to(dtype),) * 2

    def set_new_h(self, h):
        self.hidden = h

    def forward(self, x, train=False):
        # print(x.shape)
        # if not train:
        #    self.hidden = (torch.zeros((self.num_layers, 1, self.hidden_size)).to(dtype),) * 2

        if not train:
            out, self.hidden = self.lstm(x, self.hidden)
        else:
            out, _ = self.lstm(x, self.batch_hidden)

        out = self.output(out)
        return out

    def forward_action_sample(self, x):

        out, hidden = self.lstm(x, self.hidden)
        out = self.output(out)

        return out, hidden


class LSTMPSAgent(object):

    def __init__(self, state_dimension, num_actions, output_dim, hidden_size, gamma_damping=0.00, target_update=50,
                 device="cpu", learning_rate=0.01, capacity=100000,
                 eta_glow_damping=0.1, beta_softmax=0.01, batch_size=32, replay_time=100, max_len_sequence=30,
                 max_episodes=20000, seed=0):

        """
        :param state_dimension: dimension of the state that the environment outputs as an observation to the agent
        :param num_actions: total number of actions that can be taken by the agent
        :param output_dim: dimension of the output, in case of PS it should be equal to 1, since the output is the real
        value h(s,a)
        :param gamma_damping: damping parameter used in the update rule of the PS (in our case it is multiplied with the
        target h-value)
        :param target_update: update period of the target network
        :param device: "cpu"
        :param learning rate: learning rate of the optimizers
        :param capacity: maximal size of the memory deque, which stores the past (percept, reward, action)-tuples of the
         agent
        :param eta_glow_damping: "glow" hyperparameter of the PS-agent. It describes how the reward is backpropagated
         through previously stored percepts
        :param beta_softmax: steepness of the softmaxlayer, which outputs the probability distribution for the
         agent's actions.
        :param batch_size: size of the learning batch. After the memory size reaches this value, the training starts.
        :param replay_time: Since the agent by default only trains if rewarded, this value helps us increase the number
         of times the training takes place
        :param max_len_sequence: maximal length of a percept sequence
        """

        self.input_dim = state_dimension + num_actions
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.output_dim = output_dim
        self.eta_glow_damping = eta_glow_damping
        self.gamma_damping = gamma_damping
        self.beta_softmax = beta_softmax
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.capacity = capacity
        self.replay_time = replay_time
        self.batch_size = batch_size
        self.target_count = 0
        self.max_len_sequence = max_len_sequence
        self.max_episodes = max_episodes
        self.agent_type = "PS-LSTM"
        self.loss = 0
        self.seed = seed

        # Loads policy_net, target_net, and copies the polciy_net into the target net
        self.nn = NN_LSTM(self.input_dim, hidden_size, output_dim, batch_size=batch_size).to(device).to(dtype)
        self.target_nn = NN_LSTM(self.input_dim, hidden_size, output_dim, batch_size=batch_size).to(device).to(dtype)
        self.target_nn.load_state_dict(self.nn.state_dict())

        # Initializes memory as deque with maximal length capacity and short_term_memory as deque with maxlen =
        # max_len_sequence
        self.memory = deque(maxlen=capacity)

        # Initializes the optimizers and the loss function for the network
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.nn.parameters()),
                                      lr=self.learning_rate)
        self.loss_fn = F.mse_loss

        # a deque that is used to move data from one memory to another
        self.trial_data = deque()
        # the rewards of the current trial
        self.trial_rewards = torch.empty(0)
        # previous percept
        self.prev_s = None
        # previous action
        self.prev_a = None
        # number of interactions
        self.num_interactions = 0

    def reset(self):

        """
        Resets the agent's network weights
        """

        self.nn.reset_h()
        self.nn.reset_batch_h()

    def train(self):

        """
        Trains the agent's network
        :return:
        """

        # if the memory is too small, no training step is performed
        if len(self.memory) < self.batch_size:
            return 0

        self.nn.reset_batch_h()
        # print(len(self.memory))
        # periodically updates the target network by copying the values of the policy network into it
        # if target_update is 1, the two networks are always equal to each other
        self.target_count += 1
        if self.target_count % self.target_update == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        global Transition
        # randomly samples past a batch of events from the memory
        minibatch = Transition(*zip(*random.sample(self.memory, self.batch_size)))

        # creates batches of tensors for each element of Transition
        percepts = torch.cat(minibatch.percept)
        actions = torch.cat(minibatch.action)
        actions = actions.view(len(actions), 1, -1)
        rewards = torch.cat(minibatch.reward)
        rewards = rewards.view(len(rewards), 1, -1)
        # print(percepts)
        # print(actions)
        x_input = torch.cat((percepts, actions), 2)

        # clears gradients
        self.nn.zero_grad()

        # Outputs the target-h-value and the policy-h-values
        h_p = self.nn(x_input, train=True)
        h_t = self.target_nn(x_input, train=True).detach()

        # Computes the loss
        loss = F.mse_loss(h_p, h_t + rewards)
        # print(loss)

        # Clears the gradients, use backpropagation algorithm based upon the loss and runs an optimization step
        loss.backward()
        self.optim.step()

        return None

    def one_hot_encode(self, action: int) -> torch.Tensor:

        """
        One-hot encodes the action
        :param action: integer that represents the action (0,...,num_actions-1).
        :return: one-hot encoded action
        """

        a: np.ndarray = np.full(self.num_actions, 0, np.int32)
        a[action] = 1
        return torch.FloatTensor(a[np.newaxis])

    def get_action(self, percept):

        """
        Returns the action of the agent based upon the percept.
        :param percept: percept of the environment.
        :return: action of the agent.
        """

        # define output-vector of size num_actions
        h_vector = torch.zeros(self.num_actions)
        hidden_states = []

        # compute h(s,a) for every a and the given s
        for a in range(self.num_actions):
            enc_a = self.one_hot_encode(a)

            # concatenates tensors along the third dimension: I think this structure reflects
            # the basic input structure of the LSTM in pytorch: (sequence_length, batch_size, input_dimension)
            # If we concatenate along the 3rd dimension we get an input of size (state_dimension + num_actions)
            enc_a = enc_a.unsqueeze(0)
            # print(percept.unsqueeze(0))

            x_input = torch.cat([percept, enc_a], 2)
            # print(x_input.shape)

            # computes the output of the neural network without activating gradients
            with torch.no_grad():
                h_value, new_hidden = self.nn.forward_action_sample(x_input)
            # print(h_value)
            hidden_states.append(new_hidden)
            h_vector[a] = h_value

        # renormalizes h_vector to avoid exploding values in softmax layer and computes the probability distribution
        # h_vector_mod = h_vector - torch.max(h_vector)
        smax_out = F.softmax(self.beta_softmax * h_vector, dim=0)
        # print(smax_out)
        self.probability_distr = torch.distributions.Categorical(smax_out)

        # print(probability_distr)

        # computes and returns the action (the action is represented as [number])
        action = self.probability_distr.sample()
        hidden_new = hidden_states[action.item()]

        self.nn.set_new_h(hidden_new)

        # x_input_samp = torch.cat([percept, self.one_hot_encode(action).unsqueeze(0)], 2)
        # _, self.nn.hidden = self.nn.lstm(x_input_samp, self.nn.hidden)

        action = np.array([action])

        return action

    def deliberate_and_learn(self, percept, reward, done):

        """
        Deliberates and learns from the percept and the reward.
        :param percept: Percept of the environment.
        :param reward: Reward of the environment.
        :param done: Boolean that indicates whether the episode is done.
        :return: Action of the agent.
        """

        # Saves reward, done pairs into the short-term-memory
        if self.prev_s is not None:
            self.short_term_memory_push(reward, done)

        # replay training
        self.num_interactions += 1
        if self.num_interactions % self.replay_time == 0:
            if len(self.memory) > self.batch_size:
                self.train()

        # if done, short-term-memory is copied to the main memory and the short-term-memory is cleared
        # Training starts
        # support variables are cleared
        if done:  # !!!!!!!!!!!!!!!!!!!!!!!!!!
            self.memory_save()
            self.train()
            self.nn.reset_h()

            self.prev_a = None
            self.prev_s = None

            return None

        # Get a new action, updates support variables, performs and action
        action = self.get_action(percept)
        self.prev_a = self.one_hot_encode(action)
        self.prev_s = percept

        return action

    def load_state_dicts_and_memory(self, checkpoint):

        """
        Loads the state dictionaries of the neural networks and the memory.
        :param checkpoint:
        :return:
        """
        self.nn.load_state_dict(checkpoint['nn'])
        self.target_nn.load_state_dict(checkpoint['target_nn'])
        self.memory = checkpoint['memory']

    def save_checkpoint(self, args_dict, env, rewards, infidelities, circuit_length, angle_data, checkpoint_dir):

        """
        Saves the state dictionaries of the neural networks and the memory.
        :param args_dict:
        :param env:
        :param rewards:
        :param infidelities:
        :param circuit_length:
        :param angle_data:
        :param checkpoint_dir:
        :return:
        """

        torch.save(dict(nn=self.nn.state_dict(), target_nn=self.target_nn.state_dict(),
                        memory=self.memory, args_dict=args_dict, curriculum=env.curriculum, rewards=rewards,
                        infidelities=infidelities, circuit_length=circuit_length, angle_data=angle_data),
                   checkpoint_dir)

    def short_term_memory_push(self, reward, done):

        """
        Saves the reward, done pair into the short-term-memory.
        :param reward:
        :param done:
        :return:
        """

        # stores previous steps as (s,a) pairs, appends them to trial_data and appends rewards to a similar
        # trial_rewards list
        data = (self.prev_s, self.prev_a)
        self.trial_data.append(data)

        # Backpropagation of the reward ("glowing mechanism")
        #
        use_sparse_reward = True

        if use_sparse_reward:
            self.trial_rewards = torch.cat((self.trial_rewards, torch.Tensor([[reward]])))

            if done:
                if reward != 0:
                    # print(reward)
                    r_glow = max(self.trial_rewards)
                    for i in range(len(self.trial_rewards)):
                        if self.trial_rewards[i] == r_glow:
                            max_reward_index = i
                            break

                    # this gives you the length of the sequence which had the highest reward
                    lzero = len(self.trial_rewards) - max_reward_index - 1
                    # sets to zero all rewards after the maximal one
                    self.trial_rewards[max_reward_index + 1:, 0] = torch.Tensor([0] * lzero)

                    for i in range(max_reward_index + 1):
                        r_glow = r_glow * (1 - self.eta_glow_damping)
                        if r_glow < 1.e-8:
                            break
                        self.trial_rewards[max_reward_index - i] = r_glow
        else:
            r_glow = reward
            if reward != 0:
                for i in range(1, len(self.trial_rewards)):
                    r_glow = r_glow * (1 - self.eta_glow_damping)
                    if r_glow < 1.e-8:
                        break
                    # print(self.trial_rewards)
                    # print(r_glow)
                    self.trial_rewards[len(self.trial_rewards) - i] += r_glow
                    # print(self.trial_rewards)
            self.trial_rewards = torch.cat((self.trial_rewards, torch.Tensor([[reward]])))
            # print(self.trial_rewards)

        return 0

    def short_term_memory_reset(self):

        """
        Resets the short-term-memory.
        """

        # clears the trial data
        self.trial_data = deque()
        self.trial_rewards = torch.empty(0)

    def memory_save(self, factor=1):

        """
        Saves the short-term-memory into the main memory.
        :param factor: Factor that is multiplied to the rewards.
        :return:
        """

        global Transition

        # data in agent.trial_data are transfered to agent memory
        for i_f in range(factor):
            for i, data in enumerate(self.trial_data):
                data = Transition(data[0], data[1], self.trial_rewards[i])
                self.memory.append(data)

        # trial_data is cleared
        self.short_term_memory_reset()

        return


def convert_state(state, agent_type):
    """
    Converts the state into a tensor.
    :param state: State of the environment.
    :param agent_type: Type of the agent.
    :return: Tensor of the state.
    """

    if agent_type in ("PS-NN", "VanillaPG", "PPO", "REINFORCE"):
        state = torch.tensor(state.flatten()).unsqueeze(0)
    elif agent_type == "PS-LSTM":
        state = torch.tensor(state.flatten()).view(1, 1, -1)
        # print(state.shape)
    else:
        pass

    return state


def train_lstm_ps_agent(env, nn_ps_agent, cost, num_episodes, ep_start=0, checkpoint_dir=None, use_tune=False):

    """
    Trains the PS-LSTM agent.
    :param env: Reinforcement learning environment.
    :param nn_ps_agent: PS agent with neural network.
    :param cost: Cost function.
    :param num_episodes: Number of episodes.
    :param ep_start: Starting episode.
    :param checkpoint_dir: Directory to save the checkpoints.
    :param use_tune: Whether to use ray tune or not.
    :return: Tuple of rewards, infidelities, circuit lengths, and angle data.
    """

    if checkpoint_dir:

        checkpoint = torch.load(checkpoint_dir)
        rewards = checkpoint['rewards']
        infidelities = checkpoint['infidelities']
        circuit_length = checkpoint['circuit_length']
        angle_data = checkpoint['angle_data']
        nn_ps_agent.load_state_dicts_and_memory(checkpoint)
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

    cwd = os.getcwd()
    seq_data = []
    #print(num_episodes)
    #print(len(ps_agent.beta_annealing))
    assert len(nn_ps_agent.beta_annealing) == num_episodes

    for e in range(ep_start, num_episodes):

        t0 = time.time()

        state = env.reset()
        state = np.array(state)

        state = convert_state(state, nn_ps_agent.agent_type)
        nn_ps_agent.beta_softmax = nn_ps_agent.beta_annealing[e]

        episode_reward = 0
        done = False
        reward = 0

        for t in itertools.count():

            action = nn_ps_agent.deliberate_and_learn(state, reward, done)
            if cost is not None:
                next_state, reward, done, angles, infidelity = env.step(action.item(), cost)
                # Samples a transition of hybrid RL-optimization
            else:
                next_state, reward, done, angles, infidelity = env.step(action.item())
                # Samples a transition of standard RL

            # next_state = np.array(next_state)
            episode_reward += reward

            next_state = convert_state(next_state, nn_ps_agent.agent_type)
            state = next_state

            if done:
                # print(time.time() - t0)
                # print(f"Episode {e}/{num_episodes}, Reward: {episode_reward},
                # Circuit length: {len(env.gate_sequence)}")
                print(f"Agent {nn_ps_agent.seed}, PS-LSTM episode {e}/{num_episodes}, Reward: {episode_reward}, "
                      f"Length: {len(env.gate_sequence)}, Time: {time.time() - t0}")
                rewards[e] = episode_reward
                infidelities[e] = infidelity
                circuit_length[e] = len(env.gate_sequence)
                seq_data.append("-".join(env.gate_sequence) + "\n")
                # print(angles)
                angle_data[e, :len(angles), 0] = np.array(angles)
                nn_ps_agent.deliberate_and_learn(state, reward, done)
                break

        if use_tune:
            with tune.checkpoint_dir(step=e) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                with open(path, "w") as f:
                    f.write(json.dumps({"step": e}))
            tune.report(iterations=e, episode_reward=episode_reward)

        if e % 50 == 0:
            args_dict = dict(num_episodes=num_episodes, ep_start=e)
            cp_dir = os.path.join(cwd, f'checkpoint_{nn_ps_agent.agent_type}_agent_{nn_ps_agent.seed}.pth.tar')
            print(cp_dir)
            nn_ps_agent.save_checkpoint(args_dict, env, rewards, infidelities, circuit_length, angle_data, cp_dir)

    return rewards, circuit_length, seq_data, angle_data, infidelities
