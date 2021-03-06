# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py

Updated and Enhanced version of OpenAI Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
(https://github.com/openai/maddpg)
"""

import argparse
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

###########################################
#         Add modules to path             #
###########################################
# script_dir_path = os.path.abspath(os.path.dirname(__file__))
# script_dir_path = os.path.abspath(os.path.dirname(sys.argv[0]))
# split_script_dir_path = script_dir_path.split("/")
# module_parent_dir = "/".join(split_script_dir_path[:len(split_script_dir_path)-2])

# script_path = os.path.abspath(__file__)
script_path = os.path.abspath(sys.argv[0])
split_script_path = script_path.split("/")
module_parent_dir = "/".join(split_script_path[:len(split_script_path)-3])

if 'win' in sys.platform:
    split_script_path = script_path.split("\\")
    module_parent_dir = "\\".join(split_script_path[:len(split_script_path)-3])

sys.path.insert(0, module_parent_dir + '/MADDPG/')
sys.path.insert(0, module_parent_dir + '/Multi_Agent_Particle_Environment/')

from maddpg.trainer.maddpg import MADDPGAgentTrainer
from multiagent_particle_env.make_env import make_env

import maddpg.common.tf_util as tf_util


__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


def parse_args():
    """
    Parse command line arguments

    Returns:
        parser.parse_args() (argparse.Namespace): Parsed commandline arguments object
    """
    # Setup argument parser
    parser = argparse.ArgumentParser("MADDPG Reinforcement Learning experiments for Multi-Agent Particle Environments")

    # Environment
    parser.add_argument("--scenario", type=str, default="openai/simple_tag",
                        help="Name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="Maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="Number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="Number of adversaries")
    parser.add_argument("--num-fixed-adv", type=int, default=0, help="Number of adversaries following a fixed strategy")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="Policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="Policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="Number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="Number of units in the mlp")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="debug", help="Name of the experiment")
    parser.add_argument("--model-name", type=str, default="debug", help="desired name of models")
    parser.add_argument("--save-dir", type=str, default="/tmp/debug/",
                        help="Directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="Save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="Directory in which training state and model are loaded")
    parser.add_argument("--model-file", type=str, default="debug", help="Exact name of model file to restore")

    # Evaluation
    parser.add_argument("--pred-network", type=int, default=None, help="Predator network to use for action inference")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--testing", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="Number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="/tmp/debug/benchmark_files/",
                        help="Directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="/tmp/debug/learning_curves/",
                        help="Directory where plot data is saved")
    parser.add_argument("--logging", action="store_true", default=False, help="Flag to control logging of agent data")
    parser.add_argument("--log-append", type=str, default="", help="Additional string to append to log file")

    #CCM
    parser.add_argument("--perturbation", action="store_true", default=False,
                        help="Flag for controlling perturbation analysis")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation")
    parser.add_argument("--history-length", type=int, default=1, help="History/Frames in input space")
    parser.add_argument("--training-history", type=int, default=1,
                        help="Number of frames of agent history to include in training")

    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    """
    The Neural Network model used for each agent.

    Two-layer ReLU Multi Layer Perceptron (MLP) with 64 units per layer to parameterize the policies.

    The model takes as input an observation and returns values of all actions.

    Args:
        input (np.array): Observations of the world for an agent
        num_outputs (int): The number of outputs for the output layer of the model
        scope (str): The name of the scope
        reuse (boolean): Flag specifying whether to reuse the scope
        num_units (int): The number of outputs for the fully connected layers of the model
        rnn_cell (): Not currently used

    Returns:
        The outputs of the output layer of the model.

    """
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    """
    Creates the instances of learning agents.

    Args:
        env (multiagent_particle_env.environment.MultiAgentEnv): Multi-Agent Particle Environment object
        num_adversaries (int): The number of adversary agents in the environment
        obs_shape_n (list): List with the shape of the observation space of each agent
        arglist (argparse.Namespace): Parsed commandline arguments object

    Returns:
        trainers (list): A list of maddpg.trainer.maddpg.MADDPGAgentTrainer objects, one for each agent.
                         If using CCM, a list of maddpg.trainer.maddpg.MADDPGAgentTrainerCCM objects.
    """
    trainers = []
    model = mlp_model

    trainer = MADDPGAgentTrainer

    # Adversaries
    for i in range(num_adversaries):
        trainers.append(trainer(
            'agent_{}'.format(i), model, obs_shape_n, env.action_space, i, arglist, role="adversary",
            local_q_func=(arglist.adv_policy=='ddpg')))

    # Good Agents
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            'agent_{}'.format(i), model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))

    return trainers


def train(arglist):
    """
    Run MADDPG algorithm using passed in commandline arguments

    Args:
        arglist (argparse.Namespace): Parsed commandline arguments object
    """
    tf.reset_default_graph()

    if arglist.seed is not None:
        np.random.seed(arglist.seed)
        tf.set_random_seed(arglist.seed)

    # with tf_util.make_session(6):
    with tf_util.single_threaded_session():
        ###########################################
        #         Create environment              #
        ###########################################
        env = make_env(arglist.scenario, arglist=arglist, logging=arglist.logging, benchmark=arglist.benchmark)

        ###########################################
        #        Create agent trainers            #
        ###########################################
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print("Number of Adversaries: {}".format(num_adversaries))
        print('Experiment: {}. Using good policy {} and adv policy {}'.format(arglist.exp_name,
                                                                              arglist.good_policy,
                                                                              arglist.adv_policy))

        ###########################################
        #              Initialize                 #
        ###########################################
        tf_util.initialize()

        ###########################################
        #   Load previous results, if necessary   #
        ###########################################
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        # if arglist.display or arglist.restore or arglist.benchmark or arglist.load_dir is not None:
        if arglist.restore or arglist.benchmark or arglist.load_dir is not None:
            print('Loading previous state...')

            # Set model file
            if arglist.model_file == "":
                arglist.model_file = arglist.exp_name

            print("Model File: " + arglist.load_dir + arglist.model_file)
            tf_util.load_state(arglist.load_dir + arglist.model_file)

        ###########################################
        #       Create the save directory         #
        ###########################################
        if not os.path.exists(arglist.save_dir):
            os.makedirs(arglist.save_dir, exist_ok=True)

        ###########################################
        #             Set parameters              #
        ###########################################
        # Sum of rewards for all agents
        episode_rewards = [0.0]

        # This was changed so that a reward can be tracked for fixed policy agents as well as learning agents
        # Individual agent reward
        # agent_rewards = [[0.0] for _ in range(env.n)]
        agent_rewards = [[0.0] for _ in range(len(env.world.agents))]

        # Retrieve previous episode count
        try:
            prev_ep_ct = int(arglist.model_file.split("_")[-1])
        except ValueError:
            print("Starting from untrained network...")
            prev_ep_ct = 0
        ep_ct = prev_ep_ct + arglist.num_episodes

        # Sum of rewards for training curve
        final_ep_rewards = []

        # Agent rewards for training curve
        final_ep_ag_rewards = []

        # Placeholder for benchmarking info
        agent_info = [[[]]]

        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        progress = False

        # Save more often if you have fewer episodes
        arglist.save_rate = min(arglist.save_rate, arglist.num_episodes)

        ###########################################
        #                 Start                   #
        ###########################################
        print('Starting iterations...')
        while True:
            # TODO: Switch to is isinstance()
            # if type(env.world.scripted_agents[0].action) == type(None):
            #     print("Error")

            # Get action
            if arglist.pred_network is None:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            else:
                action_n = []
                for i in range(num_adversaries):
                    action_n.append(trainers[arglist.pred_network].action(obs_n[i]))

                for i in range(num_adversaries, env.n):
                    action_n.append(trainers[i].action(obs_n[i]))

            # Environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            # Logging step
            if arglist.logging:
                env.log(len(episode_rewards) + prev_ep_ct, episode_step, new_obs_n, rew_n, done_n, info_n)

            # Update information
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # Collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # Increment global step counter
            train_step += 1

            # For benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])

                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # For displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                # print("Mean Episode Reward: {}".format([np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]))
                continue

            # In testing mode, don't perform model updates
            if arglist.testing:
                if len(episode_rewards) > arglist.num_episodes:
                    print("Eval episodes: {}, "
                          "mean episode reward: {}, time: {}".format(len(episode_rewards),
                                                                     np.mean(episode_rewards[-arglist.save_rate:]),
                                                                     round(time.time()-t_start, 3)))
                    env.logger.save("State",
                                    arglist.save_dir,
                                    filename=arglist.exp_name + '_state' + '_' + str(prev_ep_ct) + arglist.log_append)
                    break
                continue

            # Update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # if len(episode_rewards) % 100 == 0 and progress:
            #     print("Episode {} Reached. Time: {}".format(len(episode_rewards), time.time() - t_start))
            #     progress = False
            # elif len(episode_rewards) % 100 != 0 and not progress:
            #     progress = True

            # Save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                # TODO: Implement some checks so that we don't overwrite old networks unintentionally?

                # Save model state
                tf_util.save_state(arglist.save_dir + arglist.exp_name + '_' + str(len(episode_rewards)+prev_ep_ct),
                                   saver=saver)

                # Print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step,
                        len(episode_rewards) + prev_ep_ct,
                        np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step,
                        len(episode_rewards) + prev_ep_ct,
                        np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(reward[-arglist.save_rate:]) for reward in agent_rewards],
                        round(time.time() - t_start, 3)))

                # Reset start time to current time
                t_start = time.time()

                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for reward in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(reward[-arglist.save_rate:]))

            # Saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)

                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

                # Log agent data for run
                env.logger.save("State", arglist.save_dir,
                                filename=arglist.exp_name + '_state' + '_' + str(len(episode_rewards) + prev_ep_ct))

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    """
    Main function

    Parses commandline arguments and calls train()
    """
    # Parse commandline arguments
    args = parse_args()

    # Start program
    train(args)
