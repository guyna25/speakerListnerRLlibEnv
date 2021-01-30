
import supersuit
from copy import deepcopy
import ray.rllib.env as rllibenv
import ray.rllib.agents.a3c.a2c as a2c
import ray
from ray.tune.registry import register_env

from pettingzoo.mpe import simple_speaker_listener_v3

alg_name = "PPO"
#config = deepcopy(get_agent_class(alg_name)._default_config)

config = deepcopy(a2c.A2C_DEFAULT_CONFIG)
# config["env"] = "simple_speaker_listner"
config["env_config"] = None
config["rollout_fragment_length"] = 20
config["num_workers"] = 5
config["num_envs_per_worker"] = 1
config["lr_schedule"] = [[0, 0.007], [20000000, 0.0000000001]]

#config["callbacks"] = MyCallbacks
config["clip_rewards"] = True
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
multiagent_dict = dict()
multiagent_policies = dict()
env = simple_speaker_listener_v3.env()
agents_name = deepcopy(env.possible_agents)
config = {
          "num_gpus": 0,
          "num_workers": 1,
          "framework": "torch",
          }
env = simple_speaker_listener_v3.env()
env = supersuit.pad_action_space_v0(env)
env = supersuit.pad_observations_v0(env)
env = rllibenv.PettingZooEnv(env)
register_env("simple_speaker_listener", lambda stam: env)

ray.init(num_gpus=0, local_mode=True)
agent = a2c.A2CTrainer(env="simple_speaker_listener", config=config)

for it in range(5):
    result = agent.train()
    print(s.format(
        it + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]
    ))
    env.reset()
