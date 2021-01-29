import os
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np

from ray.rllib.models import ModelCatalog

import ray.rllib.agents.a3c.a2c as a2c
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv
from typing import *
from ray.rllib.env.atari_wrappers import get_wrapper_by_cls
from Multi_Agent_Particle_Environment.rllib_wrapper import RLlibMultiAgentParticleEnv as SLenv
import matplotlib.pyplot as plt

# register_env("TaxiMultiAgentEnv", TaxiEnv)

config = a2c.A2C_DEFAULT_CONFIG.copy()
config["env"] = "simple_speaker_listener"
config["env_config"] = None
config["rollout_fragment_length"] = 20
config["num_workers"] = 5
config["num_envs_per_worker"] = 1
config["lr_schedule"] = [[0, 0.007], [20000000, 0.0000000001]]

# config["callbacks"] = MyCallbacks
config["clip_rewards"] = True
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
multiagent_dict = dict()
multiagent_policies = dict()
env = SLenv("simple_speaker_listener")
register_env("simple_speaker_listener",
             SLenv)
agents_name = env.get_agents_name()
# for agent in agents_name:
#     agent_entry = (
#         None,
#         env.observation_space_dict,
#         env.action_space_dict,
#         a2c.A2C_DEFAULT_CONFIG.copy(),
#     )
#     multiagent_policies[agent] = agent_entry
# multiagent_dict["policies"] = multiagent_policies
# multiagent_dict["policy_mapping_fn"] = lambda agent_id: agent_id
# config["multiagent"] = multiagent_dict

config = {'multiagent': {'policies': {agents_name[0]: (None, env.obs_space, env.action_space, a2c.A2C_DEFAULT_CONFIG.copy()),
                                      agents_name[1]: (None, env.obs_space, env.action_space, a2c.A2C_DEFAULT_CONFIG.copy())},
                         "policy_mapping_fn": lambda agent_id: agent_id},
          "num_gpus": 0,
          "num_workers": 1}

# config = {'multiagent': {'policies': {'taxi_1': (None, env.obs_space, env.action_space, {'gamma': TAXI1_GAMMA}),
#                                       'taxi_2': (None, env.obs_space, env.action_space, {'gamma': TAXI2_GAMMA})
#                                       },
#                          "policy_mapping_fn": lambda taxi_id: taxi_id},
#           "num_gpus": NUM_GPUS,
#           "num_workers": NUM_WORKERS}

ray.init(num_gpus=0, local_mode=True)
agent = a2c.A2CTrainer(config=config, env="simple_speaker_listener")

for it in range(5):
    result = agent.train()
    print(s.format(
        it + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]
    ))
# tune.run(
    # a2c.A2CTrainer,
    # config=config,
    # stop={"timesteps_total": 100},
    # checkpoint_at_end=True,
# )
