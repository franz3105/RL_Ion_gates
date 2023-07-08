import random
import itertools
import time
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from rl_agents.lstm_ps import convert_state
from ray import tune
from collections import deque, namedtuple, OrderedDict

Transition = namedtuple('Transition', ('percept', 'action',
                                       'reward'))  # we use namedtuple as a data structure to store events with their
# properties (percept, action, reward)

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double


def gradClamp(model, clip=5):  # in case we need to clamp gradients
    nn.utils.clip_grad_norm(model.parameters(), clip)


class DQN(nn.Module):
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

    dropout_rate: During training, randomly zeroes some of the elements of the input
    tensor with probability dropout using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_

    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.0, num_layers=0):
        super(DQN, self).__init__()
        self.norm = nn.LayerNorm((input_dim)).to(dtype)

        self.input = nn.Linear(input_dim, hidden_dim).to(dtype)
        nn.init.kaiming_uniform_(self.input.weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.input.bias, a=0, mode='fan_in', nonlinearity='relu')
        # Kaiming initialization
        # of weights for feedforward networks: "Surpassing Human-Level Performance on ImageNet Classification",
        # Kaiming He

        hidden_dict = OrderedDict()
        for i in range(num_layers):
            hidden_dict["lin{}".format(i)] = nn.Linear(hidden_dim, hidden_dim)
            hidden_dict["relu{}".format(i)] = nn.ReLU()

        self.hidden = nn.Sequential(hidden_dict).to(dtype)

        self.output = nn.Linear(hidden_dim + 1, output_dim).to(dtype)
        nn.init.constant_(self.output.weight, 1.0)
        nn.init.constant_(self.output.bias, 1.0)
        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False

        # add connection from first layer to last layer with gradients
        # last layer only of weights 1 and without gradients

        self.last_sum_layer = nn.Linear(input_dim, output_dim).to(dtype)
        # additional neuron resulting as a weighted sum of all the inputs

        self.dropout = nn.Dropout(p=dropout_rate).to(dtype)

    def forward(self, x):
        # 1 normalization layer
        out = self.norm(x)
        # compute sum of the initial input
        final_x_sum = self.last_sum_layer(out)

        # print(self.input.weight)
        # hidden Linear-ReLU layers
        out = F.relu(self.input(x))
        out = self.hidden(out)
        # concatenated tensors (dimension: hidden_dim +1)
        out = torch.cat((out, final_x_sum), 1)

        out = F.relu(self.dropout(self.output(out)))
        return out


class DeepPSAgent(object):
    """
    state_dimension: dimension of the state that the environment outputs as an observation to the agent

    num_actions: total number of actions that can be taken by the agent

    output_dim: dimension of the output, in case of PS it should be equal to 1, since the output is the real value h(s,a)

    gamma_damping: damping parameter used in the update rule of the PS (in our case it is multiplied with the target h-value)

    target_update: update period of the target network

    device: "cpu" ("gpu" is not implemented)

    hidden_dim =

    learning rate: learning rate of the optimizers

    capacity: maximal size of the memory deque, which stores the past (percept, reward, action)-tuples of the agent

    eta_glow_damping: "glow" hyperparameter of the PS-agent. It describes how the reward is backpropagated through
     previously stored percepts

    beta_softmax = steepness of the softmaxlayer, which outputs the probability distribution for the agent's actions.

    batch_size = size of the learning batch. After the memory size reaches this value, the training starts.

    replay_time = Since the agent by default only trains if rewarded, this value helps us increase the number of times
     the training takes place


    """

    def __init__(self, state_dimension, num_actions, policy_type="softmax", hidden_dim=128, dropout_rate=0.0,
                 target_update=50, device="cpu", learning_rate=0.01, capacity=20000000,
                 eta_glow_damping=0.02, beta_softmax=0.01, gamma_damping=0.00, batch_size=64, replay_time=50,
                 num_layers=2, seed=0, constant_nn_dim=False, **kwargs):
        # super(DeepPSAgent, self).__init__(state_dimension, num_actions)
        # add the beta parameter as np.linespace(beta_min, beta_max, N)
        # torch.manual_seed(seed)
        self.input_dim = (int(state_dimension / num_actions) + 1) * num_actions
        self.num_actions = num_actions
        self.eta_glow_damping = eta_glow_damping
        self.policy_type = policy_type
        self.beta_softmax = beta_softmax
        self.gamma_damping = gamma_damping
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.capacity = capacity
        self.replay_time = replay_time
        self.batch_size = batch_size
        self.target_count = 0
        self.probability_distr = None
        self.max_num_actions = max(self.num_actions, 10)
        self.nn_dim = state_dimension + self.max_num_actions
        self.seed = seed

        # Loads policy_net, optimizers, target_net, and copies the polciy_net into the target net
        self.dqn = None
        self.target_dqn = None
        self.dqn = DQN(self.nn_dim, hidden_dim, num_layers=num_layers).to(device)
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dqn.parameters()),
                                      lr=self.learning_rate, amsgrad=True)
        self.loss_fn = F.mse_loss

        self.target_dqn = DQN(self.nn_dim, hidden_dim, num_layers=num_layers).to(device).to(dtype)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Initializes memory as deque with maximal length capacity and short_term_memory as deque with maxlen =
        # max_len_sequence
        self.memory = deque(maxlen=capacity)
        # Initializes the optimizers and the loss function for the network

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
        self.agent_type = "PS-NN"

    def one_hot_encode(self, action):

        """
        one-hot encodes the action.
        :param action: (np.ndarray) action to be encoded.
        :return: (torch.tensor) one-hot encoded action.
        """
        # every action corresponts to a unit vector
        a = torch.full((self.max_num_actions,), 0, dtype=torch.int32)
        # print(a)
        a[action[0]] = 1
        # print(a)
        return a.unsqueeze(0).to(dtype)

    def encode_state(self, percept):
        """
        Encodes the percept into a tensor of dimension (1, input_dim).
        :param percept: (np.ndarray) percept to be encoded.
        :return: (torch.tensor) encoded percept.
        """

        feat_vec = np.zeros((self.num_actions, int(self.input_dim / self.num_actions)))
        for i_p, p in enumerate(percept[0]):
            feat_vec[p, i_p] = 1
        return torch.tensor([feat_vec]).to(dtype)

    def pad_percept(self, percept):

        """
        Pads the percept with zeros to match the dimension of the network input.
        :param percept: (torch.tensor) percept to be padded.
        :return: (torch.tensor) padded percept.
        """
        zero_pad = torch.zeros(percept.shape[0], self.nn_dim - self.num_actions)
        zero_pad[:percept.shape[0], :percept.shape[1]] = percept
        return zero_pad

    def train(self):

        """
        Trains the network. It samples a batch of events from the memory and performs a training step.
        :return:
        """

        # if the memory is too small, no training step is performed
        if len(self.memory) < self.batch_size:
            return 0

        # periodically updates the target network by copying the values of the policy network into it
        # if target_update is 1, the two networks are always equal to each other
        self.target_count += 1
        if self.target_count % self.target_update == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        global Transition
        # randomly samples past a batch of events from the memory
        minibatch = Transition(*zip(*random.sample(self.memory, self.batch_size)))

        # creates batches of tensors for each element of Transition
        percepts = torch.cat(minibatch.percept).to(dtype)
        actions = torch.cat(minibatch.action).to(dtype)
        rewards = torch.cat(minibatch.reward).to(dtype)
        rewards = rewards.view(len(rewards), 1).to(dtype)

        x_input = torch.cat((percepts, actions), 1).to(dtype)

        # Outputs the target-h-value and the policy-h-values
        h_p = self.dqn(x_input)
        h_t = self.target_dqn(x_input)

        # Computes the loss
        loss = F.mse_loss(h_p, h_t + rewards - self.gamma_damping * h_t)
        # print(loss)
        # Clears the gradients, use backpropagation algorithm based upon the loss and runs an optimization step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return None

    def get_action(self, percept):
        """
        Chooses an action according to the policy networks h-value predictions.
        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment
        Returns:
            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([m])
        """
        # output vector of size

        h_vector = torch.zeros(self.num_actions)
        for a in range(self.num_actions):
            enc_a = self.one_hot_encode([a])
            x_input = torch.cat([percept, enc_a], dim=1).to(dtype)
            with torch.no_grad():
                h_value = self.dqn(x_input)[0]
            h_vector[a] = h_value

        # h_vecotor is scaled to avoid exploding values in the exponential function
        # h_vector = h_vector - torch.max(h_vector)
        # h_vector[h_vector < 1.e-10] = 0.

        # print(h_vector_mod)
        # output probability distribution
        # h_vector_mod = h_vector - np.max(h_vector)
        smax_out = F.softmax(self.beta_softmax * h_vector, dim=0)
        # print(smax_out)
        probability_distr = torch.distributions.Categorical(smax_out)
        # probability_distr = torch.exp(self.beta_softmax * h_vector) / torch.sum(torch.exp(self.beta_softmax *
        # h_vector)) print(probability_distr) probability_distr[probability_distr < 1.e-10] = 0.
        self.probability_distr = probability_distr
        # print(probability_distr)
        # action = torch.random.sa(range(self.num_actions), p=self.probability_distr)
        # action = np.array([action])
        action = probability_distr.sample()
        # print(action)

        return np.array([action])

    def deliberate(self, percept, reward):
        """
        Chooses an action according to the policy networks h-value predictions.
        :param percept: (torch.Tensor) the current percept issued by the environment.
        :param reward: (torch.Tensor) the reward issued by the environment.
        :return: action: (torch.Tensor) the action that is to be performed as a tensor of torch.size([m]).
        """
        # if an environment state was given as an observation to the agent, store the reward
        if self.prev_s is not None:
            self.memory_push(reward)

        action = self.get_action(percept)
        self.prev_a = self.one_hot_encode(action)
        self.prev_s = percept

        return action

    def learn(self):
        """
        Performs a learning step.
        :return: None
        """
        # if done, saves the events in the main memory, trains and clears silly variables
        self.memory_save()
        self.train()

        self.prev_a = None
        self.prev_s = None

    def deliberate_and_learn(self, percept, reward, done):

        """
        Chooses and action and performs a learning step.
        :param percept: (torch.Tensor) the current percept issued by the environment.
        :param reward: (torch.Tensor) the reward issued by the environment.
        :param done: (bool) whether the episode is over or not.
        :return: (torch.Tensor) the action that is to be performed as a tensor of torch.size([m]).
        """

        # if an environment state was given as an observation to the agent, store the reward
        if self.prev_s is not None:
            self.memory_push(reward)

        # replay training
        self.num_interactions += 1
        # if self.num_interactions % self.replay_time == 0:
        # print(self.memory)
        # self.memory_save()
        # self.train()

        # if done, saves the events in the main memory, trains and clears silly variables
        if done:
            self.memory_save()
            self.train()

            self.prev_a = None
            self.prev_s = None

            return None

        # get action, update silly variables and returns action
        action = self.get_action(percept)
        self.prev_a = self.one_hot_encode(action)
        self.prev_s = percept

        return action

    def memory_push(self, reward):

        """
        Stores the previous state and action in the trial_data list and the reward in the trial_rewards list.
        :param reward: (torch.Tensor) the reward issued by the environment.
        :return: None
        """

        # stores previous steps as (s,a) pairs, appends them to trial_data and appends rewards to a similar
        # trial_rewards list
        data = (self.prev_s, self.prev_a)
        self.trial_data.append(data)

        # Backpropagation of rewards through the reward list that was previously stored
        r_glow = reward
        if reward != 0:
            for i in range(1, len(self.trial_rewards)):
                r_glow = r_glow * (1 - self.eta_glow_damping)
                if r_glow < 1.e-20:
                    break
                # print(self.trial_rewards)
                # print(r_glow)
                self.trial_rewards[len(self.trial_rewards) - i] += r_glow
                # print(self.trial_rewards)
        self.trial_rewards = torch.cat((self.trial_rewards, torch.Tensor([[reward]])))

        return 0

    def load_state_dicts_and_memory(self, checkpoint):

        """
        Loads the state dictionaries of the neural networks and the memory.
        :param checkpoint: (dict) the checkpoint dictionary.
        :return: None
        """

        self.dqn.load_state_dict(checkpoint['nn'])
        self.target_dqn.load_state_dict(checkpoint['target_nn'])
        self.memory = checkpoint['memory']

    def save_checkpoint(self, args_dict, env, rewards, infidelities, circuit_length, angle_data, checkpoint_dir):

        """
        Saves the state dictionaries of the neural networks and the memory.
        :param args_dict: (dict) the checkpoint dictionary.
        :param env: (Environment) the environment.
        :param rewards: (list) the list of rewards.
        :param infidelities: (list) the list of infidelities.
        :param circuit_length: (list) the list of circuit lengths.
        :param angle_data: (list) the list of angle data.
        :param checkpoint_dir: (str) the directory where the checkpoint is to be saved.
        :return:
        """

        torch.save(dict(nn=self.dqn.state_dict(), target_nn=self.target_dqn.state_dict(),
                        memory=self.memory, args_dict=args_dict, curriculum=env.curriculum, rewards=rewards,
                        infidelities=infidelities, circuit_length=circuit_length, angle_data=angle_data),
                   checkpoint_dir)

    def memory_save(self):

        """
        Saves the trial data in the main memory.
        :return:
        """

        global Transition

        # Data saved in trial data are transfered to the main memory (the one used for training)
        for i, data in enumerate(self.trial_data):
            data = Transition(data[0], data[1], self.trial_rewards[i])
            self.memory.append(data)

        # Clears the trial lists
        self.trial_data = deque()
        self.trial_rewards = torch.empty(0)

        return 0


def train_nn_ps_agent(env, nn_ps_agent, cost, num_episodes, ep_start=0, checkpoint_dir=None, use_tune=False):
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
    # print(num_episodes)
    # print(len(ps_agent.beta_annealing))
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
                print(f"Agent {nn_ps_agent.seed}, NN-PS episode {e}/{num_episodes}, Reward: {episode_reward}, "
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
