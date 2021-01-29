# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
alternate_policies.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np
import math
import matplotlib.pyplot as plt#Erin
import random#Erin
import copy#Erin
from scipy.optimize import newton, newton_krylov, brentq#James
from numpy import sinh, sqrt#James


__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


def distance_minimizing_fixed_strategy(agent, world):
    """
    Distance-minimizing fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect prey agents
    prey = []
    for other in world.policy_agents:
        if not other.adversary:
            prey.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    my_prey = prey[0]
    x = my_prey.state.p_pos[0] - agent.state.p_pos[0]
    y = my_prey.state.p_pos[1] - agent.state.p_pos[1]

    x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))

    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def random_fixed_strategy(agent, world):
    """
    Random fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    # Set random action
    random_act = np.random.random()*2*np.pi
    agent.action.u[0] = np.cos(random_act)
    agent.action.u[1] = np.sin(random_act)

    # Scale random action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def spring_fixed_strategy(agent, world):
    """
    Random fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    k = 10
    F = np.zeros((2,))

    for other in world.agents:
        dis = np.linalg.norm((other.state.p_pos - agent.state.p_pos))
        if other.adversary and other != agent:
            # F += k * (np.linalg.norm(other.state.p_pos - agent.state.p_pos) - 0.5) * \
            #      (other.state.p_pos - agent.state.p_pos)
            F += k * (dis - 0.5) * ((other.state.p_pos - agent.state.p_pos) / dis)
        if not other.adversary and other != agent:
            # F += 0.4 * k * 1 / (np.linalg.norm(other.state.p_pos - agent.state.p_pos)) * \
            #      (other.state.p_pos - agent.state.p_pos)
            F += 0.2 * k * (1 / dis) * (other.state.p_pos - agent.state.p_pos)

    F = F / np.linalg.norm(F)

    # Scale spring action by acceleration
    agent.action.u = agent.accel * F

    return agent.action


def spring_fixed_strategy_2(agent, world):
    """
    Random fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    k = 10
    F = np.zeros((2,))

    for other in world.agents:
        dis = np.linalg.norm((other.state.p_pos - agent.state.p_pos))
        if other.adversary and other != agent:
            # F += k * (np.linalg.norm(other.state.p_pos - agent.state.p_pos) - 0.5) * \
            #      (other.state.p_pos - agent.state.p_pos)
            F += k * (dis - 0.5) * ((other.state.p_pos - agent.state.p_pos) / dis)
        # if not other.adversary and other != agent:
            # F += 0.4 * k * 1 / (np.linalg.norm(other.state.p_pos - agent.state.p_pos)) * \
            #      (other.state.p_pos - agent.state.p_pos)
            # F += 0.2 * k * (1 / dis) * (other.state.p_pos - agent.state.p_pos)

    F = F / np.linalg.norm(F)

    # Scale spring action by acceleration
    agent.action.u = agent.accel * F

    return agent.action


def sheep_fixed_strategy(agent, world):
    """
    Sheep fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    threat_rad = 0.5
    pred_threats = []
    for other in world.policy_agents:
        if other.adversary:
            vec = np.zeros(world.dimension_position)
            vec[0] = other.state.p_pos[0] - agent.state.p_pos[0]
            vec[1] = other.state.p_pos[1] - agent.state.p_pos[1]
            d = np.linalg.norm(vec)
            if d <= threat_rad:
                pred_threats.append(vec)

    if len(pred_threats) > 0:
        agent.action.u = -1 * (np.sum(pred_threats, 0)) * (1 / np.linalg.norm(np.sum(pred_threats, 0)))
    else:
        agent.action.u = np.zeros(world.dimension_position)
        a = np.random.random() * 2 * np.pi
        agent.action.u[0] = np.cos(a)
        agent.action.u[1] = np.sin(a)

        # Scale sheep action by acceleration
        agent.action.u = agent.accel * agent.action.u

    return agent.action


def evader_fixed_strategy(agent, world):
    """
    Evader distance-minimizing fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect predator agents
    preds = []
    for i, other in enumerate(world.policy_agents):
        if other.adversary:
            preds.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    x = []
    y = []
    for pred in preds:
        xpred = pred.state.p_pos[0] - agent.state.p_pos[0]
        ypred = pred.state.p_pos[1] - agent.state.p_pos[1]

        scale = np.linalg.norm(np.array([xpred, ypred]))
        scale = np.exp(-4 * scale)

        x.append(scale * xpred)
        y.append(scale * ypred)

    for i, obs in enumerate(world.landmarks):
        xobs = obs.state.p_pos[0] - agent.state.p_pos[0]
        yobs = obs.state.p_pos[1] - agent.state.p_pos[1]

        scale = np.linalg.norm(np.array([xobs, yobs]))

        # scale = np.max([np.exp(-2 * scale), 1e-1])
        scale = np.exp(-10 * (scale - 3))
        # print("Scale: {}".format(scale))

        xobs, yobs = xobs / np.linalg.norm(np.array([xobs, yobs])), yobs / np.linalg.norm(np.array([xobs, yobs]))

        x.append(scale * xobs)
        y.append(scale * yobs)

    x, y = np.mean(np.array([x, y]), 1)

    x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))

    agent.action.u[0] = -x_n
    agent.action.u[1] = -y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def trapper_fixed_strategy(agent, world):
    """
    Trapper distance-minimizing fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect prey agents
    prey = []
    for i, other in enumerate(world.policy_agents):
        if not other.adversary:
            prey.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    my_prey = prey[0]
    xp = my_prey.state.p_pos[0] - agent.state.p_pos[0]
    yp = my_prey.state.p_pos[1] - agent.state.p_pos[1]
    r = 1.0 / 3.0

    # world.landmarks[0].state.p_pos = agent.state.p_pos

    if np.linalg.norm(np.array([xp, yp])) < r:
        if agent.counter < 10:
            x, y = xp, yp
            agent.counter += 1
        else:
            x, y = np.array([0.0, 0.0]) - agent.state.p_pos
    else:
        x, y = np.array([0.0, 0.0]) - agent.state.p_pos
        if np.linalg.norm(np.array([x, y])) < r:
            agent.counter = 0
        if np.linalg.norm(np.array([x, y])) < agent.size:
            x, y = 0.0, 0.0

    if np.linalg.norm(np.array([x, y])) != 0.0:
        x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))
    else:
        x_n, y_n = 0.0, 0.0

    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


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
    L1, L2 = 0.49, 0.49
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


def double_pendulum_perturbation_strategy(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy

    Updates agent's state and zero's out action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    agent.action.u = np.zeros(world.dimension_position)
    agent.state.state = state

    return agent.action


def double_pendulum_perturbation_strategy_2(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy.

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    agent.action.u = np.array([L1 * np.cos(d_state[1]), L1 * np.sin(d_state[1])]) + \
                     np.array([L2 * np.cos(d_state[3]), L2 * np.sin(d_state[3])])
    agent.state.state = state

    return agent.action


def double_pendulum_perturbation_strategy_3(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy.

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    agent.action.u = a1 + a2
    agent.state.state = state

    return agent.action


def double_pendulum_perturbation_strategy_4(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy

    Updates agent's state and zero's out action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    x = agent.state.p_pos[0]
    y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

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


def perturbation_strategy_old(agent, world):
    """
    Strategy for perturbing an agent's policy

    Updates agent's state and zero's out action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        # agent.state.p_pos = 0.49 * np.array([np.sin(th[0]),
        #                                      -np.cos(th[0])]) + 0.45 * np.array([np.sin(th[1]), -np.cos(th[1])])
        # agent.state.p_vel = 0.49 * np.array([np.cos(w[0]),
        #                                      -np.cos(w[0])]) + 0.45 * np.array([np.cos(w[1]), -np.cos(w[1])])
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    # world.landmarks[-1].state.p_pos = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    agent.action.u = np.zeros(world.dimension_position)
    agent.state.state = state

    return agent.action


def perturbation_strategy_2_old(agent, world):
    """
    Strategy for perturbing an agent's policy

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        # agent.state.p_pos = 0.49 * np.array([np.sin(th[0]),
        #                                      -np.cos(th[0])]) + 0.45 * np.array([np.sin(th[1]), -np.cos(th[1])])
        # agent.state.p_vel = 0.49 * np.array([np.cos(w[0]),
        #                                      -np.cos(w[0])]) + 0.45 * np.array([np.cos(w[1]), -np.cos(w[1])])
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    # world.landmarks[-1].state.p_pos = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    agent.action.u = np.array([L1 * np.cos(d_state[1]), L1 * np.sin(d_state[1])]) + \
                     np.array([L2 * np.cos(d_state[3]), L2 * np.sin(d_state[3])])
    agent.state.state = state

    return agent.action


def perturbation_strategy_3_old(agent, world):
    """
    Strategy for perturbing an agent's policy

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        # agent.state.p_pos = 0.49 * np.array([np.sin(th[0]),
        #                                      -np.cos(th[0])]) + 0.45 * np.array([np.sin(th[1]), -np.cos(th[1])])
        # agent.state.p_vel = 0.49 * np.array([np.cos(w[0]),
        #                                      -np.cos(w[0])]) + 0.45 * np.array([np.cos(w[1]), -np.cos(w[1])])
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    # world.landmarks[-1].state.p_pos = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    agent.action.u = a1 + a2
    agent.state.state = state

    return agent.action
    
def bearing_strategy(agent,world):
    """
    Bearing-angle fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    def toDeg(x):#For convenience in debugging
        return(x*180/math.pi)
    plots = False#Set to True for debugging
    
    # Collect predatory and prey agents (prey borrowed from distance_minimizing_fixed_strategy)
    prey = []
    for other in world.policy_agents:
        if not other.adversary:
            prey.append(other)

    my_prey = prey[0]#because there might be more than one, but then you ignore the rest?? Pick the closest one?
    
    #If evader has no velocity, return action unchanged. This is used for the first timestep.
    if np.all(my_prey.state.p_vel == np.array([0,0])):
        print("No velocity - returning unchanged action")
        return(agent.action)
    else:
        # Zero out agent action (borrowed from distance_minimizing_fixed_strategy)
        original_action = copy.deepcopy(agent.action)#In case I need to return unchanged as filler, shouldn't be needed for final version
        agent.action.u = np.zeros(world.dimension_position) #world.dimension_position = 2, not sure why this is necessary

    #Now come some weird names that help me with testing
    #Pursuer
    x_p = agent.state.p_pos[0]
    y_p = agent.state.p_pos[1]
    #agent_max_speed = 1
    V_p = agent.max_speed #Max speed
    V_P = agent.max_speed #Actual speed. Assume max speed, not actual speed from agent.stat.p_vel Might need to check that not None, not sure this needs to be normed
    #Evader
    x_e = my_prey.state.p_pos[0]
    y_e = my_prey.state.p_pos[1]
    V_E = np.linalg.norm(my_prey.state.p_vel) #Actual speed. Assuming p_vel is change in [x,y], new position is p_pos + p_vel * dt (where dt is timestep, .1)
    #Setting the stage
    r = math.sqrt((x_p - x_e)**2 + (y_p - y_e)**2)#Distance between pursuer and evader
    if r == 0:#Capture
        #Plot for testing
        if plots:
            plt.plot([x_p], [y_p], 'bo')#pursuer
            plt.plot([x_e], [y_e], 'ro')#evader
            plt.plot([x_e+my_prey.state.p_vel[0]],[y_e+my_prey.state.p_vel[1]], 'r*')
            plt.plot([x_p,x_e], [y_p,y_e], 'black')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axhline(y=0, color='gray')
            plt.axvline(x=0, color='gray')
            plt.title("r=0")
            plt.grid()
            plt.show()
        return(original_action)
    if (y_e - y_p < 1): r *= -1 #This might make calculating intersection easier?
    nu = V_E/V_P #Mike said to use actual speeds 
    
    #Mike's math magic to calculate unambiguous phi
    r_E_P_3D = np.array([x_p - x_e, y_p - y_e, 0])#Vector from evader to pursuer, 3D
    V_E_3D = np.array([my_prey.state.p_vel[0], my_prey.state.p_vel[1], 0])
    temp = np.cross(r_E_P_3D, V_E_3D)
    phi = math.atan2(temp[2],np.dot(r_E_P_3D[0:2],V_E_3D[0:2]))
    
    vers = ""
    
    if nu >= 1: #nu=V_E/V_P, faster evader
        phi_star = math.asin(1/nu)
    else:#Faster pursuer
        phi_star = phi#To avoid div by zero error. phi_star should be irrelevant when V_E<V_p anyway.
        
    if V_E < V_p or abs(phi) <= phi_star:#Mike had these separate
        vers = "Capture possible"
        #print(vers)
        theta = -math.asin((V_E/V_P)*math.sin(phi)) #Equation 4
    else:
        vers = "Capture impossible"
        #print(vers)
        def no_capture_theta(theta, phi=phi,V_P=V_P,V_E=V_E):
            return(math.sqrt(1-V_P**2/V_E**2)*math.sin(theta) - math.sin(theta+phi))
        if phi >= 0:
            theta = -brentq(no_capture_theta, 0, math.pi, maxiter=500)
        else:
            theta = -brentq(no_capture_theta, -math.pi, 0, maxiter=500)

    rho = math.pi-abs(theta)-abs(phi) #angle opposite r
    if rho == 0:#r=0, already collided
        print("Rho = 0, this shouldn't happen anymore")
        #return(original_action)#Perhaps not best action
    p = abs((r/math.sin(rho))*math.sin(phi)) #distance between pursuer and intersection
    #e = abs((r/math.sin(rho))*math.sin(theta))#distance between evader and intersection
    #print(abs(r/math.sin(rho)) == abs(e/math.sin(theta)) == abs(p/math.sin(phi)) )#Checking, law of sines
    
    m_r = (y_p - y_e)/(x_p - x_e)
    #Find slope of p using theta and difference between theta and x axis
    if m_r != None:
        m_p = math.tan(math.atan(m_r)+theta) #Calculate slope of p using theta plus angle between theta and x axis
    else:
        m_p = math.pi/2 - math.tan(math.atan(0)+theta)#Doubel check, I don't remember what exactly this is doing

    #OK, where do p and e meet? 
    #Calculate the intersection's offset from pursuer
    #Distance between pursuer and intersection **2 = x_change**2 + y_change**2
    #p**2 = x_change**2 + y_change**2 #m_p*x_change = y_change (change from p, flat not angled for slope to work)
    #p**2 = x_change**2 + m_p_test**2 * x_change**2 
    x_change = math.sqrt(abs(p**2 / (1+m_p**2))) #We'll change sign later if necessary
    x_c = x_p + x_change#So the location of the intersection is the location of the pursuer + the x and y changes
    y_change = m_p * x_change
    y_c = y_p + y_change
    #[x_c,y_c]#This is the point where pursuer and evader would meet
        
    #Use change from pursuer to find new pursuer velocity (if p were of length V_p, what would it's x and y components be?)
    x_change_scaled = V_P*(1/p)*x_change #It's V_P/sqrt(1+m_p**2), but it should be same sign as x_change, so I'm using a roundabout way
    y_change_scaled = m_p*x_change_scaled 
    new_vel = np.array([x_change_scaled,y_change_scaled])#The pursuer's new velocity is along the slope of p, scaled by speed

    tried_sign = False
    sf = 1#-1 to change the direction along m_p that pursuer will go
    while True:
        # print("Looping...")
        A = np.array([x_e+my_prey.state.p_vel[0],y_e+my_prey.state.p_vel[1]])
        B = np.array([x_e,y_e])
        C = np.array([x_c,y_c])
        #If points are not on the same line (and so their area isn't 0), either we chose the wrong intersection angle or we picked the wrong sign for sqrt when calculating pursuer's x change
        #Try both separate and together, they should ultimately be on same line
        bad_area = abs(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1])	+ C[0]*(A[1]-B[1])) > 0.01 #Should be 0
        bad_order = abs(np.linalg.norm(C-A) - np.linalg.norm(A-B) - np.linalg.norm(C-B)) < 0.01 #Should be greater than 0
        if not tried_sign and (bad_area or bad_order):
            sf = -1
            tried_sign = True
        else:
            break
        #Redo some calculations - OK, where do p and e meet? 
        #Calculate the intersection's offset from pursuer
        #Distance between pursuer and intersection **2 = x_change**2 + y_change**2
        #p**2 = x_change**2 + y_change**2 #m_p*x_change = y_change (change from p, flat not angled for slope to work)
        #p**2 = x_change**2 + m_p_test**2 * x_change**2 
        x_change = sf * math.sqrt(abs(p**2 / (1+m_p**2))) #We'll change sign later if necessary
        x_c = x_p + x_change#So the location of the intersection is the location of the pursuer + the x and y changes
        y_change = m_p * x_change
        y_c = y_p + y_change
        #[x_c,y_c]#This is the point where pursuer and evader would meet
        
        #Use change from pursuer to find new pursuer velocity (if p were of length V_p, what would it's x and y components be?)
        x_change_scaled = V_P*(1/p)*x_change #It'sV_P/(1+m_p**2), but it should be same sign as x_change, so I'm using a roundabout way
        y_change_scaled = m_p*x_change_scaled 
        new_vel = np.array([x_change_scaled,y_change_scaled])#The pursuer's new velocity is along the slope of p, scaled by speed

    #Set action to pursuer change in x and y, scale action by acceleration (borrowed from distance_minimizing_fixed_strategy)
    agent.action.u = agent.accel * new_vel
    #Plot to check
    if plots:
        plt.plot([x_p], [y_p], 'bo')#pursuer
        plt.plot([x_p,x_c], [y_p,y_c], 'b--')
        plt.plot([x_p+new_vel[0]],[y_p+new_vel[1]], 'b*')
        plt.plot([x_e], [y_e], 'ro')#evader
        plt.plot([x_e,x_c], [y_e,y_c], 'r')
        plt.plot([x_e+my_prey.state.p_vel[0]],[y_e+my_prey.state.p_vel[1]], 'r*')
        plt.plot([x_p,x_e], [y_p,y_e], 'black')
        plt.plot([x_c], [y_c], 'gx')#intersection
        plt.plot([x_p+agent.action.u[0]], [y_p+agent.action.u[1]], 'y*')#Action?
        plt.title(vers+'\nphi='+str(toDeg(phi))+'\ntheta='+str(toDeg(theta)))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axhline(y=0, color='gray')
        plt.axvline(x=0, color='gray')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid()
        #plt.show()
        plt.savefig('C:/Users/erin.g.zaroukian/Desktop/Multi-Agent-Behaviors-Team/testPlots/'+str(random.randint(0,100000000)) + '.png')#Sorry, in scripts for now...
        plt.clf()

    return(agent.action)

        
