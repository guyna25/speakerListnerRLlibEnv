# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_tag_fixed_min.py

Predator-prey

Communication: No
Competitive:   Yes

Good agents (green) are faster and want to avoid being hit by adversaries (red).
Adversaries are slower and want to hit good agents. Obstacles block the screen edges.

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

from array import array
import numpy as np

from multiagent_particle_env.alternate_policies import distance_minimizing_fixed_strategy
from multiagent_particle_env.core import World, Agent, Landmark
from multiagent_particle_env.scenario import BaseScenario

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


def _double_pendulum(state):
    """
    Differential dynamics of a double pendulum

    Args:
        state (multiagent_particle_env.core.AgentState.state): State element from Agent state object
              or
              (list) Agent state 4-element list

    Returns:
        Updated agent state causing agent to act as a double pendulum
    """
    L1, L2 = 0.5, 0.5
    G = 9.8
    M1, M2 = 1.0, 1.0

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(del_) * np.cos(del_)
    dydx[1] = (M2 * L1 * state[1] * state[1] * np.sin(del_) * np.cos(del_) +
               M2 * G * np.sin(state[2]) * np.cos(del_) +
               M2 * L2 * state[3] * state[3] * np.sin(del_) -
               (M1 + M2) * G * np.sin(state[0])) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (-M2 * L2 * state[3] * state[3] * np.sin(del_) * np.cos(del_) +
               (M1 + M2) * G * np.sin(state[0]) * np.cos(del_) -
               (M1 + M2) * L1 * state[1] * state[1] * np.sin(del_) -
               (M1 + M2) * G * np.sin(state[2])) / den2

    return dydx


def _double_pendulum_perturbation_strategy(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    x = agent.state.p_pos[0]
    y = agent.state.p_pos[1]
    L1, L2 = 0.5, 0.5

    try:
        state = agent.state.state
    except:
        if (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) >= 1.0:
            th2 = 0.0
        elif (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) <= -1.0:
            th2 = np.pi
        else:
            th2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))

        th1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
        w1, w2 = np.random.random(1)[0]**2, np.random.random(1)[0]**2
        state = np.array([th1, w1, th2, w2])

    if all(state == 0.0):
        if (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) >= 1.0:
            th2 = 0.0
        elif (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) <= -1.0:
            th2 = np.pi
        else:
            th2 = np.arccos((x**2 + y**2 - L1**2 - L2**2)/(2*L1*L2))

        th1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
        w1, w2 = np.random.random(1)[0]**2, np.random.random(1)[0]**2
        state = np.array([th1, w1, th2, w2])

    # Scaled double pendulum state
    d_state = _double_pendulum(state)/40.
    state = state + d_state

    new_x = L2*np.sin(state[2]) + L1*np.sin(state[0])
    new_y = -L2*np.cos(state[2]) - L1*np.cos(state[0])

    agent.state.p_pos[0] = new_x
    agent.state.p_pos[1] = new_y

    agent.state.state = state
    agent.action.u = np.zeros(world.dimension_position)

    agent.action.u[0] = d_state[1]
    agent.action.u[1] = d_state[3]

    return agent.action


class Scenario(BaseScenario):
    """
    Define the world, reward, and observations for the scenario.
    """

    def make_world(self, args):
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Create world and set properties
        world = World()
        world.dimension_communication = 2
        world.log_headers = ["Agent_Type", "Fixed", "Perturbed", "X", "Y", "dX", "dY", "fX", "fY", "Collision"]
        num_good_agents = 1
        num_adversaries = 0
        num_fixed_adv = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(num_fixed_adv + i)
            agent.adversary = True if i < num_adversaries else False
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.max_speed = 1.0 if agent.adversary else 1.3

        # Add fixed policy agents
        fixed_agents = [Agent() for i in range(num_fixed_adv)]
        for i, fixed in enumerate(fixed_agents):
            fixed.name = 'agent {}'.format(i)
            fixed.adversary = True
            fixed.accel = 3.0
            fixed.collide = True
            fixed.max_speed = 1.0
            fixed.silent = True
            fixed.size = 0.075
            if args.perturbation:
                fixed.action_callback = _double_pendulum_perturbation_strategy
                fixed.is_perturbed_policy = True
                args.perturbation = False
            else:
                fixed.action_callback = distance_minimizing_fixed_strategy
                fixed.is_fixed_policy = True
        world.agents = fixed_agents + world.agents

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.boundary = False
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2

        # Make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set properties for agents
        for agent in world.agents:
            agent.color = np.array([52., 239., 233.]) / 256 if not agent.adversary else np.array([0.85, 0.35, 0.35])

        # Set properties for scripted agents
        for scripted_agent in world.scripted_agents:
            scripted_agent.color = np.array([239., 180., 52.]) / 256

        # Set properties for landmarks
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)
            agent.state.state = np.zeros((4,))

        # Set random initial states for landmarks and boundary landmarks
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dimension_position)
                landmark.state.p_vel = np.zeros(world.dimension_position)

    def benchmark_data(self, agent, world):
        """
        Returns data for benchmarking purposes.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is an adversary:
                (int) The total number of collisions between all good agents
            Else:
                (int) Zero
        """
        if agent.adversary:
            # Determine total number of collisions with all good agents
            collisions = 0
            for a in self.good_agents(world):
                if world.is_collision(a, agent):
                    collisions += 1

            return collisions
        else:
            return 0

    def good_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        """
        Returns all agents that are adversaries in a list.

        Returns:
            (list) All the agents in the world that are adversaries.
        """
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world, shaped=False):
        """
        Reward is based on prey agent not being caught by predator agents.

        Good agents are negatively rewarded if caught by adversaries and for exiting the screen.

        Adversaries are rewarded for collisions with good agents.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped (boolean): Specifies whether to use shaped reward, adds distance based increase and decrease.

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        if agent.adversary:
            return self.adversary_reward(agent, world, shaped)
        else:
            return self.agent_reward(agent, world, shaped)

    def agent_reward(self, agent, world, shaped):
        """
        Good agents are negatively rewarded if caught by adversaries and for exiting the screen.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped (boolean): Specifies whether to use shaped reward, increased for increased distance from adversaries.

        Returns:
            (float) Total agent reward, based on avoiding collisions and staying within the screen.
        """
        reward = 0
        adversaries = self.adversaries(world)

        # Reward can optionally be shaped (increased reward for increased distance from adversary)
        if shaped:
            for adv in adversaries:
                reward += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))

        # Determine collisions and assign penalties
        if agent.collide:
            for adv in adversaries:
                if world.is_collision(adv, agent):
                    reward -= 10

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            reward -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        return reward

    def adversary_reward(self, agent, world, shaped):
        """
        Adversaries are rewarded for collisions with good agents.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped (boolean): Specifies whether to use shaped reward, decreased for increased distance from good agents.

        Returns:
            (float) Total agent reward, based on avoiding collisions and staying within the screen.
        """
        # Adversaries are rewarded for collisions with agents
        reward = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        # Reward can optionally be shaped (decreased reward for increased distance from agents)
        if shaped:
            for adv in adversaries:
                reward -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])

        # Determine collisions and assign rewards
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if world.is_collision(ag, adv):
                        reward += 10

        return reward

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with the velocity of the agent, the position of the agent,
                       distance to all landmarks in the agent's reference frame,
                       distance to all other agents in the agent's reference frame,
                       and the velocities of the good agents.
        """
        # Get positions all landmarks that are not boundary markers in this agent's reference frame
        landmarks_pos = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Communication, positions, and velocities of all other agents in this agent's reference frame
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

            # Only store velocities for good agents
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        # pad = []
        # if not agent.adversary:
        #     pad.append(array('d', [0, 0]))

        debug = False
        if debug:
            print("### AGENT {} ###".format(agent.name))
            # len = 2
            print("agent.state.p_vel: {}".format(agent.state.p_vel))
            # len = 2
            print("agent.state.p_pos: {}".format(agent.state.p_pos))
            # len = 0
            print("landmarks_pos: {}".format(landmarks_pos))
            # len = 2 x 3
            print("other_pos: {}".format(other_pos))
            # len = 2
            print("other_vel: {}".format(other_vel))

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmarks_pos + other_pos + other_vel + pad)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmarks_pos + other_pos + other_vel)

    def logging(self, agent, world):
        """
        Collect data for logging.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (list) Data for logging
        """
        # Log elements
        agent_type = ""
        fixed = agent.is_fixed_policy
        perturbed = agent.is_perturbed_policy
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]
        dx = agent.state.p_vel[0]
        dy = agent.state.p_vel[1]
        fx = agent.action.u[0]
        fy = agent.action.u[1]
        collision = 0

        # Check for collisions
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent in good_agents:
            agent_type = "good"
            for adv in adversaries:
                if world.is_collision(agent, adv):
                    collision += 1
        elif agent in adversaries:
            agent_type = "adversary"
            for ga in good_agents:
                if world.is_collision(agent, ga):
                    collision += 1
        else:
            collision = "N/A"

        log_data = [agent_type, fixed, perturbed, x, y, dx, dy, fx, fy, collision]

        return log_data
