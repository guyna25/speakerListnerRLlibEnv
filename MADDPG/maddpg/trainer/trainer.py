# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
trainer.py

Contains the base skeleton for trainers using MADDPG

Updated and Enhanced version of OpenAI Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
(https://github.com/openai/maddpg)
"""

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


class AgentTrainer(object):
    """
    Defines the base AgentTrainer object.
    """

    def __init__(self):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents, steps):
        raise NotImplemented()
