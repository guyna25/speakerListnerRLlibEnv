# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
policy.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import argparse
import os
import sys

from multiagent_particle_env.environment import MultiAgentEnv
from multiagent_particle_env.logger import Log, Logger
from multiagent_particle_env.policy import InteractivePolicy

import multiagent_particle_env.scenarios as scenarios

sys.path.insert(1, os.path.join(sys.path[0], '..'))

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='openai/simple.py', help='Path of the scenario Python script.')
    parser.add_argument("--logging", action="store_true", default=False, help="flag to control logging of agent data")
    args = parser.parse_args()

    # Load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()

    # Create world
    world = scenario.make_world()

    # Set up logger
    logger = Logger(args.logging)

    # Create multi-agent environment
    env = MultiAgentEnv(world, logger, scenario.reset_world, scenario.reward, scenario.observation,
                        shared_viewer = False)

    # Render call to create viewer window (necessary only for interactive policies)
    env.render()

    # Create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]

    # Execution loop
    obs_n = env.reset()

    while True:
        # Query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))

        # Step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # Render all agent views
        env.render()

        # Display rewards
        # for agent in env.world.agents:
        #     print(agent.name + " reward: {}".format(env._get_reward(agent)))
