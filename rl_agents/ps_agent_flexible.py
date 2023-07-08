# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Katja Ried, implementing ideas from 

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400

and 

'Projective Simulation for Classical Learning Agents: A Comprehensive Investigation'
Julian Mautner, Adi Makmal, Daniel Manzano, Markus Tiersch & Hans J. Briegel
New Generation Computing, Volume 33, Issue 1, pp 69-114 (2015) doi:10.1007/s00354-015-0102-0
"""

# This code requires the following packages
import os
import itertools
import time
import numpy as np
import torch
import json

from ray import tune


## PROJECTIVE SIMULATION AGENT (more sophisticated version)

class FlexiblePSAgent(object):

    def __init__(self, num_actions, gamma_damping, eta_glow_damping, policy_type, beta_softmax, seed=0):
        """Initialize the basic PS agent. Arguments: 
        - num_actions: integer >=1, 
        - gamma_damping: float between 0 and 1, controls forgetting/damping of h-values
        - eta_glow_damping: float between 0 and 1, controls the damping of glow; setting this to 1 effectively switches off glow
        - policy_type: string, 'standard' or 'softmax'; toggles the rule used to compute probabilities from h-values
        - beta_softmax: float >=0, probabilities are proportional to exp(beta*h_value). If policy_type != 'softmax', then this is irrelevant.
        """
        self.num_percepts = 0
        self.num_actions = num_actions
        self.gamma_damping = gamma_damping  # damping parameter controls forgetting, gamma
        self.eta_glow_damping = eta_glow_damping  # damping of glow, eta
        self.policy_type = policy_type
        self.beta_softmax = beta_softmax
        self.min_reward = 0
        self.seed = 0

        self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64)
        self.r_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64)

        # dictionary of raw percepts
        self.percept_dict = {}
        self.agent_type = "PS"

    def percept_preprocess(self, observation):  # preparing for creating a percept
        """Takes a percept of any immutable form -- numbers, strings or tuples thereof --
        or lists and arrays (which are flattened and converted to tuples), 
        checks whether it has been encountered before,
        updates num_percepts, percept_dict, h_matrix and g_matrix if required and
        and returns a single integer index corresponding to the percept."""
        # MEMO: the list of immutable types and the handling of mutable types (notably arrays) may need expanding
        # in order to ensure that we cover all 
        # Try to turn the observation into an immutable type
        if type(observation) in [str, int, bool, float, np.float64,
                                 tuple]:  # This list should contain all relevant immutable types
            dict_key = observation
        elif type(observation) == list:
            dict_key = tuple(observation)
        elif type(observation) == np.ndarray:
            dict_key = tuple(observation.flatten())
        else:
            raise TypeError('Observation is of a type not supported as dictionary key. You may be able to add a way '
                            'of handling this type.')

        if dict_key not in self.percept_dict.keys():
            self.percept_dict[dict_key] = self.num_percepts
            self.num_percepts += 1
            # add column to hmatrix, gmatrix
            self.h_matrix = np.append(self.h_matrix, np.ones([self.num_actions, 1]), axis=1)
            self.r_matrix = np.append(self.r_matrix, np.zeros([self.num_actions, 1]), axis=1)

        return self.percept_dict[dict_key]

    def save_checkpoint(self, args_dict, env, rewards, infidelities, circuit_length, angle_data, checkpoint_dir):

        return

    def load_checkpoint(self, checkpoint_path):

        return

    def deliberate_and_learn(self, observation, reward, done):
        """Given an observation and a reward (from the previous interaction), this method
        updates the h_matrix, chooses the next action and records that choice in the g_matrix.
        Arguments: 
            - observation: any immutable object (as specified for percept_preprocess), 
            - reward: float
        Output: action, represented by a single integer index."""
        self.h_matrix = self.h_matrix + self.gamma_damping * (self.h_matrix - 1.) + reward * self.r_matrix
        percept = self.percept_preprocess(observation)
        action = np.random.choice(self.num_actions, p=self.probability_distr(percept))  # deliberate once
        self.r_matrix = (1 - self.eta_glow_damping) * self.r_matrix
        self.r_matrix[action, percept] = 1
        # learning and forgetting

        return action

    def probability_distr(self, percept):

        if self.policy_type == 'softmax':
            h_vector = self.beta_softmax * self.h_matrix[:, percept]
            h_vector_mod = h_vector - np.amax(h_vector)
            probability_distr = np.exp(h_vector_mod) / np.sum(np.exp(h_vector_mod))
        else:
            raise NotImplementedError("This function is not implemented")
        return probability_distr


def train_ps_agent(env, ps_agent, cost, num_episodes, ep_start=0, checkpoint_dir=None, use_tune=False):
    """
    Trains the PS-LSTM agent.
    :param env: Reinforcement learning environment.
    :param ps_agent: PS agent with neural network.
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
        ps_agent.load_state_dicts_and_memory(checkpoint)
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
    assert len(ps_agent.beta_annealing) == num_episodes

    for e in range(ep_start, num_episodes):

        t0 = time.time()

        state = env.reset()
        state = np.array(state)

        ps_agent.beta_softmax = ps_agent.beta_annealing[e]

        episode_reward = 0
        done = False
        reward = 0

        for t in itertools.count():

            action = ps_agent.deliberate_and_learn(state, reward, done)
            if cost is not None:
                next_state, reward, done, angles, infidelity = env.step(action, cost)
                # Samples a transition of hybrid RL-optimization
            else:
                next_state, reward, done, angles, infidelity = env.step(action)
                # Samples a transition of standard RL

            # next_state = np.array(next_state)
            episode_reward += reward

            state = next_state

            if done:
                # print(time.time() - t0)
                # print(f"Episode {e}/{num_episodes}, Reward: {episode_reward},
                # Circuit length: {len(env.gate_sequence)}")
                print(f"Agent {ps_agent.seed}, PS episode {e}/{num_episodes}, Reward: {episode_reward}, "
                      f"Length: {len(env.gate_sequence)}, Time: {time.time() - t0}")
                rewards[e] = episode_reward
                infidelities[e] = infidelity
                circuit_length[e] = len(env.gate_sequence)
                seq_data.append("-".join(env.gate_sequence) + "\n")
                # print(angles)
                angle_data[e, :len(angles), 0] = np.array(angles)
                ps_agent.deliberate_and_learn(state, reward, done)
                break

        if use_tune:
            with tune.checkpoint_dir(step=e) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                with open(path, "w") as f:
                    f.write(json.dumps({"step": e}))
            tune.report(iterations=e, episode_reward=episode_reward)

        if e % 50 == 0:
            args_dict = dict(num_episodes=num_episodes, ep_start=e)
            cp_dir = os.path.join(cwd, f'checkpoint_{ps_agent.agent_type}_agent_{ps_agent.seed}.pth.tar')
            print(cp_dir)
            ps_agent.save_checkpoint(args_dict, env, rewards, infidelities, circuit_length, angle_data, cp_dir)

    return rewards, circuit_length, seq_data, angle_data, infidelities
