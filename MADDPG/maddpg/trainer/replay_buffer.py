# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
replay_buffer.py

Updated and Enhanced version of OpenAI Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
(https://github.com/openai/maddpg)
"""

from collections import deque

import numpy as np
import random

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Prioritized Replay buffer.

        Args:
            size (int): Max number of transitions to store in the buffer. When the buffer
                        overflows the old memories are dropped.
        """
        # self._storage = []
        self._storage = deque([])
        self._count = deque([])
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        """
        Compute the length of the replay buffer object

        Returns:
            The length of the replay buffer storage list.
        """
        return len(self._storage)

    def clear(self):
        """
        Clears the replay buffer
        """
        # self._storage = []
        self._storage = deque([])
        self._count = deque([])
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        Add a transition data element to replay buffer

        Args:
            obs_t (np.array): Observations of the world for an agent
            action (list): Action for an agent
            reward (float): Reward for an agent
            obs_tp1 (np.array): New observations of the world for an agent
            done (): Done for an agent
        """
        # Original implementation

        # data = (obs_t, action, reward, obs_tp1, done)
        #
        # if self._next_idx >= len(self._storage):
        #     self._storage.append(data)
        # else:
        #     self._storage[self._next_idx] = data
        # self._next_idx = (self._next_idx + 1) % self._maxsize

        data = (obs_t, action, reward, obs_tp1, done)

        if self._maxsize >= len(self._storage):
            self._count.append(self._next_idx)
            self._storage.append(data)
        else:
            self._count.append(self._next_idx)
            self._count.popleft()

            self._storage.append(data)
            self._storage.popleft()

        self._next_idx += 1

    def _encode_sample(self, idxes):
        """
        Sample experiences for the given indices.

        Args:
            idxes (list): List of transition indexes to encode

        Returns:
            (tuple) Experience samples from replay buffer for given indexes.
                    (np.array(observations), np.array(actions), np.array(rewards),
                    np.array(new_observations), np.array(dones))
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_sample_histories(self, idxes, history):
        """
        Sample experiences for the given indices and history.

        Args:
            idxes (list): List of transition indexes to encode
            history (int): Number of histories to collect

        Returns:
            (tuple) Experience samples from replay buffer for given indexes.
                    (np.array(observations), np.array(actions), np.array(rewards),
                    np.array(new_observations), np.array(dones),
                    np.array(observations_history), np.array(actions_history), np.array(rewards_history),
                    np.array(new_observations_history), np.array(dones_history))
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        obses_t_h, actions_h, rewards_h, obses_tp1_h, dones_h = [], [], [], [], []
        for i in idxes:
            # Current
            if i == len(self._storage):
                data = self._storage[i - 1]
            else:
                # Original implementation
                data = self._storage[i]

            # Original implementation
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

            # History
            if history != 0:
                temp = list(self._storage)
                if (i - history) > 0:
                    data = temp[i - history:i]
                else:
                    data = temp[i]
                    data = history * [data]

                obs_t_h, action_h, reward_h, obs_tp1_h, done_h = np.array([]), np.array([]), [], np.array([]), []
                for dat in data:
                    obs_t_h = np.concatenate((obs_t_h, dat[0]), 0)
                    action_h = np.concatenate((action_h, dat[1]), 0)
                    reward_h.append(dat[2])
                    obs_tp1_h = np.concatenate((obs_tp1_h, dat[3]), 0)
                    done_h.append(dat[4])

                obses_t_h.append(np.array(obs_t_h, copy=False))
                actions_h.append(np.array(action_h, copy=False))
                rewards_h.append(reward_h)
                obses_tp1_h.append(np.array(obs_tp1_h, copy=False))
                dones_h.append(done_h)

        return (np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones),
                np.array(obses_t_h), np.array(actions_h), np.array(rewards_h), np.array(obses_tp1_h), np.array(dones_h))

    def make_index(self, batch_size):
        """
        Create list of (n) random indexes, where n = batch_size

        Args:
            batch_size (int): How many transitions to sample.

        Returns:
            (list) List of random indexes
        """
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        """
        Create a randomly shuffled list of the last (n) transitions, where n = batch_size

        Args:
            batch_size (int): How many transitions to sample.

        Returns:
            (list) Random shuffled list of (n) indexes.
        """
        # Original implementation

        # idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        # np.random.shuffle(idx)
        # return idx

        idx = [len(self._storage) - 1 - i for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def make_episode_index(self, batch_ep_size, ep_len, shuffle=True):
        """
        Create lists of (n) random indexes and (m) random indexes, where n = batch_ep_size and m = ep_len

        Args:
            batch_ep_size (int):
            ep_len (int):
            shuffle (boolean): Flag for whether to shuffle indexes

        Returns:
            (tuple) Random shuffled list of (n) indices and (n) ccm episode indices
                    (idx, ep_idx)

        """
        ep_start_locs = [i for i, x in enumerate(self._count) if x % ep_len == 0][0:-1]
        eps = [x for x in range(len(ep_start_locs))][::-1]

        if shuffle:
            np.random.shuffle(eps)
        eps = eps[0:batch_ep_size]
        eps = [ep_start_locs[x] for x in eps]

        # eps = [ep_start_locs[random.randint(0,len(ep_start_locs)-1)] for _ in range(batch_ep_size)]
        # np.random.shuffle(eps)
        # temp = int(len(ep_start_locs)/2)
        # eps = ep_start_locs[temp:temp+batch_ep_size]
        idx = [x for ep in eps for x in range(ep, ep + ep_len)]

        if shuffle:
            np.random.shuffle(idx)
        ep_idx = [[x for x in range(ep, ep + ep_len)] for ep in eps]

        return idx, ep_idx

    def sample_index(self, idxes, history=0):
        """
        Sample experiences for the given indices.

        Args:
            idxes (list): List of indexes to collect samples
            history (int): Training history

        Returns:
            (tuple) Batch of experience samples from replay buffer for given indexes.
                    (np.array(observations), np.array(actions), np.array(rewards),
                    np.array(new_observations), np.array(dones))
        """
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Args:
            batch_size (int): How many transitions to sample.

        Returns:
            (tuple) Batch of experience samples from replay buffer.
                    (np.array(observations), np.array(actions), np.array(rewards),
                    np.array(new_observations), np.array(dones))

                    or

            (tuple) All experience samples from replay buffer.
                    (np.array(observations), np.array(actions), np.array(rewards),
                    np.array(new_observations), np.array(dones))
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))

        return self._encode_sample(idxes)

    def collect(self):
        """
        Collect all samples in replay buffer

        Returns:
            (tuple) All experience samples from replay buffer.
                    (np.array(observations), np.array(actions), np.array(rewards),
                    np.array(new_observations), np.array(dones))
        """
        return self.sample(-1)
