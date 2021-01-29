# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
maddpg.py

Updated and Enhanced version of OpenAI Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
(https://github.com/openai/maddpg)
"""

import numpy as np
import tensorflow as tf

from maddpg.common.distributions import make_pdtype
from maddpg.trainer.replay_buffer import ReplayBuffer
from maddpg.trainer.trainer import AgentTrainer

import maddpg.common.tf_util as tf_util

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


def discount_with_dones(rewards, dones, gamma):
    """
    Discounts agent rewards with dones from scenario dones funtion.

    Args:
        rewards (list): Rewards for all agents
        dones (list): Dones for all agents
        gamma (float): Scalar for reward

    Returns:
         Discounted rewards
    """
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + (gamma * r)
        r = r * (1. - done)
        discounted.append(r)

    return discounted[::-1]


def make_update_exp(vals, target_vals):
    """
    Update target network values using polyak averaging (exponentially decaying average).

    Args:
        vals (tf.Variable): Network variables
        target_vals (tf.Variable): Target network variables

    Returns
        Updated target network values
    """
    # Polyak coefficient for Polyak-averaging of the target network
    polyak = 1.0 - 1e-2

    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        # Exponentially decaying average
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)

    return tf_util.function([], [], updates=[expression])


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None):
    """
    Policy learning guided by Q-value

    Args:
        make_obs_ph_n (tf.placeholder): Placeholder for the observation space of all agents
        act_space_n (list): A list of the action spaces for all agents
        p_index (int): Agent index number
        p_func (function): MLP Neural Network model for the agent.
        q_func (function): MLP Neural Network model for the agent.
        optimizer (function): Network Optimizer function
        grad_norm_clipping (float): Value by which to clip the norm of the gradient
        local_q_func (boolean): Flag for using local q function
        num_units (int): The number outputs for the layers of the model
        scope (str): The name of the scope
        reuse (boolean): Flag specifying whether to reuse the scope

    Returns:
        act (function): Action function for retrieving agent action.
        train (function): Training function for P network
        update_target_p (function): Update function for updating P network values
        p_debug (dict): Contains 'p_values' and 'target_act' of the P network
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # Set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = tf_util.scope_vars(tf_util.absolute_scope_name("p_func"))

        # Wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = tf_util.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = tf_util.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = tf_util.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = tf_util.function([obs_ph_n[p_index]], p)

        # Target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]),
                          scope="target_p_func", num_units=num_units)
        target_p_func_vars = tf_util.scope_vars(tf_util.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = tf_util.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None):
    """
    Q-Learning

        make_obs_ph_n (tf.placeholder): Placeholder for the observation space of all agents
        act_space_n (list): A list of the action spaces for all agents
        q_index (int): Agent index number
        q_func (function): MLP Neural Network model for the agent.
        optimizer (function): Network Optimizer function
        grad_norm_clipping (float): Value by which to clip the norm of the gradient
        local_q_func (boolean): Flag for using local q function
        num_units (int): The number outputs for the layers of the model
        scope (str): The name of the scope
        reuse (boolean): Flag specifying whether to reuse the scope

    Returns:
        train (function): Training function for Q network
        update_target_q (function): Update function for updating Q network values
        q_debug (dict): Contains 'q_values' and 'target_q_values' of the Q network
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # Set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = tf_util.scope_vars(tf_util.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # Viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        # loss = q_loss + 1e-3 * q_reg
        loss = q_loss

        optimize_expr = tf_util.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = tf_util.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = tf_util.function(obs_ph_n + act_ph_n, q)

        # Target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = tf_util.scope_vars(tf_util.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = tf_util.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):
    """
    Agent Trainer using MADDPG Algorithm
    """
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, role="", local_q_func=False):
        """
        Args:
            name (str): Name of the agent
            model (function): MLP Neural Network model for the agent.
            obs_shape_n (tf.placeholder): Placeholder for the observation space of all agents
            act_space_n (list): A list of the action spaces for all agents
            agent_index (int): Agent index number
            args (argparse.Namespace): Parsed commandline arguments object
            role (str): Role of the agent i.e. adversary
            local_q_func (boolean): Flag for using local q function
        """
        # super(MADDPGAgentTrainer, self).__init__()

        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args

        # Set up observation space placeholder
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(tf_util.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = 30 # args.batch_size * args.max_episode_len TODO: Change back
        self.replay_sample_index = None

    def action(self, obs):
        """
        Retrieves action for agent from the P network given the observations

        Args:
            obs (np.array): Observations of the world for an agent

        Returns:
            Action for an agent
        """
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        """
        Store transition in the replay buffer.

        Args:
            obs (np.array): Observations of the world for an agent
            act (list): Action for an agent
            rew (float): Reward for an agent
            new_obs (np.array): New observations of the world for an agent
            done (): Done for an agent
            terminal (boolean): Flag for whether the final episode has been reached.
        """
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        """
        Reset replay_sample_index to None.
        """
        self.replay_sample_index = None

    def update(self, agents, steps):
        """
        Update agent networks

        Args:
            agents (list): List of MADDPGAgentTrainer objects
            steps (int): Current training step

        Returns:
            (list) Training loss for the agents
                   [q_loss, p_loss, mean_target_q, mean_reward, mean_target_q_next, std_target_q]
        """
        # Replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return

        # Only update every 100 steps
        if not steps % 100 == 0:
            return

        # Collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        self_index = self.replay_sample_index
        for i in range(self.n):
            index = agents[i].replay_buffer.make_index(self.args.batch_size)
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(self_index)

        # Train Q Network
        num_sample = 1
        target_q = 0.0
        target_q_next = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # Train P Network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
