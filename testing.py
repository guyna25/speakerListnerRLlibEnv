import os
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np

from ray.rllib.models import ModelCatalog
from copy import deepcopy
from numpy import float32
import os
from supersuit import normalize_obs_v0, dtype_v0, color_reduction_v0

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from pettingzoo.butterfly import pistonball_v3

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
from pettingzoo.mpe import simple_speaker_listener_v3
# register_env("TaxiMultiAgentEnv", TaxiEnv)

def env_creator(config):
    env = simple_speaker_listener_v3.env()
    env = dtype_v0(env, dtype=float32)
    env = color_reduction_v0(env, mode="R")
    env = normalize_obs_v0(env)
    return env
alg_name = "PPO"
#config = deepcopy(get_agent_class(alg_name)._default_config)

config = deepcopy(a2c.A2C_DEFAULT_CONFIG.copy())
# config["env"] = "simple_speaker_listner"
# config["env_config"] = None
# config["rollout_fragment_length"] = 20
# config["num_workers"] = 5
# config["num_envs_per_worker"] = 1
# config["lr_schedule"] = [[0, 0.007], [20000000, 0.0000000001]]
#
# config["callbacks"] = MyCallbacks
# config["clip_rewards"] = True
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
multiagent_dict = dict()
multiagent_policies = dict()
env = SLenv('simple_speaker_listener')
agents_name = env.get_agents_name()
config = {'multiagent': {'policies': {agents_name[0]: (None, env.obs_space, env.action_space, a2c.A2C_DEFAULT_CONFIG.copy()),
                                      agents_name[1]: (None, env.obs_space, env.action_space, a2c.A2C_DEFAULT_CONFIG.copy())},
                         "policy_mapping_fn": lambda agent_id: agent_id},
          "num_gpus": 0,
          "num_workers": 1,
          #"env": SLenv
          }
register_env("simple_speaker_listner", lambda config: PettingZooEnv(env_creator(config)))

test_env = PettingZooEnv(env_creator({}))
obs_space = test_env.observation_space
act_space = test_env.action_space
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
    test_env.reset()
