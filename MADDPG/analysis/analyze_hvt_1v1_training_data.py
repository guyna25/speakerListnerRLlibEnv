# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_hvt_1v1_training_data.py

Creates plots for reward and loss data collected during training.
"""

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns

mpl.rcParams['agg.path.chunksize'] = 10000

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'


def plot_loss(arglist, data, agent, plot_dir_path, plot_title="Loss"):
    """
    Plot a given list of training rewards of model and save the plot

    Args:
        arglist (argparse.Namespace): Parsed commandline arguments object
        data (pandas.core.series.Series): Agent training loss for a model
        agent (str): Agent name
        plot_dir_path (str): Directory where to save plots
        plot_title (str): Title for the plot
    """
    y_axis_label = 'Loss'
    x_axis_label = '# of Loss sample'

    plt.plot(data)
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(plot_title)

    # Save plot
    plot_file_name = "{}_{}_training_loss.pdf".format(arglist.model, agent)
    plt.savefig(os.path.join(plot_dir_path, plot_file_name), dpi=600, bbox_inches='tight')

    plot_file_name = "{}_{}_training_loss.png".format(arglist.model, agent)
    plt.savefig(os.path.join(plot_dir_path, plot_file_name), dpi=600, bbox_inches='tight')

    # Close plot to free up space, needed for the for loop of all agents.
    plt.close()


def plot_reward(arglist, data, agent, plot_dir_path, plot_title="Rewards"):
    """
    Plot a given list of training rewards of model and save the plot

    Args:
        arglist (argparse.Namespace): Parsed commandline arguments object
        data (list): List of training rewards for a model
        agent (str): Agent name
        plot_dir_path (str): Directory where to save plots
        plot_title (str): Title for the plot
    """
    y_axis_label = 'Mean Reward'
    x_axis_label = '# of 1000 Episode Batches'

    plt.plot(data)
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(plot_title)

    # Save plot
    plot_file_name = "{}_{}_training_reward.pdf".format(arglist.model, agent)
    plt.savefig(os.path.join(plot_dir_path, plot_file_name), dpi=600, bbox_inches='tight')

    plot_file_name = "{}_{}_training_reward.png".format(arglist.model, agent)
    plt.savefig(os.path.join(plot_dir_path, plot_file_name), dpi=600, bbox_inches='tight')

    # Close plot to free up space, needed for the for loop of all agents.
    plt.close()


def parse_args():
    """
    Parse command line arguments

    Returns:
        parser.parse_args() (argparse.Namespace): Parsed commandline arguments object
    """
    # Setup argument parser
    parser = argparse.ArgumentParser("Create a 2D histogram from recorded agent position data")

    # Optional arglist arguments
    # parser.add_argument("--no-save", action="store_true", help="Flag for not saving the graph")
    # parser.add_argument("--show", action="store_true", help="Flag for showing the graph")

    # Required arglist arguments
    required_named_args = parser.add_argument_group('Required named arguments')
    required_named_args.add_argument("--model", type=str, default=None, help="Name of model to plot", required=True)
    required_named_args.add_argument("--num-episodes", type=int, default=200000, help="Number of training epsiodes", required=True)
    required_named_args.add_argument("--data-path", type=str, default=None, help="Absolute path of data folder",
                                     required=True)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function

    Parses commandline arguments and calls plot functions
    """
    ###########################################
    #      Parse command line arguments       #
    ###########################################
    args = parse_args()

    ###########################################
    #          Import model data              #
    ###########################################
    model_data_path = os.path.join(args.data_path, args.model)

    # Load pickled rewards
    rewards = pickle.load(open(os.path.join(model_data_path, "{}_rewards.pkl".format(args.model)), 'rb'))
    ag_rewards = pickle.load(open(os.path.join(model_data_path, "{}_agrewards.pkl".format(args.model)), 'rb'))

    # Slice agent rewards into attacker and defender
    # list[start:stop:step]
    attacker_rewards = ag_rewards[0::2]
    defender_rewards = ag_rewards[1::2]

    # Load agent p_loss
    attacker_loss = pd.read_csv(os.path.join(model_data_path, "agent_0_loss_episodes_{}.csv".format(args.num_episodes)))
    defender_loss = pd.read_csv(os.path.join(model_data_path, "agent_1_loss_episodes_{}.csv".format(args.num_episodes)))

    plot_dir = os.path.join(model_data_path, "plots")
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    ###########################################
    #          Plot loss and rewards          #
    ###########################################
    # Plot rewards
    plot_reward(args, rewards, "combined", plot_dir,
                plot_title="{} Combined Mean Training Reward".format(args.model))
    plot_reward(args, attacker_rewards, "attacker", plot_dir,
                plot_title="{} Attacker Mean Training Reward".format(args.model))
    plot_reward(args, defender_rewards, "defender", plot_dir,
                plot_title="{} Defender Mean Training Reward".format(args.model))

    # Plot loss
    plot_loss(args, attacker_loss['p_loss'], "attacker", plot_dir,
              plot_title="{} Attacker Training Loss".format(args.model))
    plot_loss(args, defender_loss['p_loss'], "defender", plot_dir,
              plot_title="{} Defender Training Loss".format(args.model))
