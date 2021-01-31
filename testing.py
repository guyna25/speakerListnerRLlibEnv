import supersuit
from copy import deepcopy
from ray.rllib.env import PettingZooEnv
import ray.rllib.agents.a3c.a2c as a2c
import ray
from ray.tune.registry import register_env
from pettingzoo.mpe import simple_speaker_listener_v3

alg_name = "PPO"
env_name = "name"
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
def create_env(args):
    if args[env_name] == "simple_speaker_listener":
        env = simple_speaker_listener_v3.env()
        env = supersuit.pad_action_space_v0(env)
        env = supersuit.pad_observations_v0(env)
        return env
register_env("simple_speaker_listener", lambda config: PettingZooEnv(create_env(config)))
config = deepcopy(a2c.A2C_DEFAULT_CONFIG)
config.update({
            "num_gpus": 0,
            "lr_schedule": [[0, 0.007], [20000000, 0.0000000001]],
            #"num_workers": 5,
            "framework": "torch",
            "env_config": {"name": "simple_speaker_listener"},
            "clip_rewards": True,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 20,
          })
ray.init(num_gpus=0, local_mode=True)
agent = a2c.A2CTrainer(env="simple_speaker_listener", config=config)

#env.reset()
for it in range(5):
    result = agent.train()
    print(s.format(
        it + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]
    ))

